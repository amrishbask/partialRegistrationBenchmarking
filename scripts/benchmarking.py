import os
import argparse
import tqdm
import json
from datetime import datetime
import glob
import subprocess
from time import time

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import tqdm

def make_beforeAfter_plot(before, after, graph_dir):
    before = np.array(before).flatten()
    after = np.array(after).flatten()
    indices = np.where(after > np.max(before))
    colors = np.ones((after.shape[0], 3))*np.array([0,0,1])
    colors[indices] = np.array([1,0,0])
    num_greater_10 = np.sum(np.where(after > 0.01, 1, 0))
    print("percentage above 10mm = ", num_greater_10/ len(after))
    after = np.where(after > np.max(before),np.max(before), after)
    plt.figure(figsize=[20, 20])
    for x, y, c in zip(before, after, colors):
        plt.scatter(x, y, color=c)
    # plt.scatter(before, after, s=10, alpha=0.8)
    min = np.min([np.min(before), np.min(after)])
    max = np.max([np.max(before), np.max(after)])
    plt.plot([0, np.max(before)], [0, np.max(before)], "k--")
    plt.plot([0.0, np.max(before)], [0.01, 0.01], "r--")
    # plt.axis("equal")
    plt.xlim([0, np.min([0.1, np.max(before)])])
    plt.ylim([0, np.min([0.1, np.max(before)])])
    plt.yticks(np.array(range(0, int(1000*np.min([0.1, max])), 2))/1000)
    plt.xlabel("Max seam offset before PR")
    plt.ylabel("Max seam offset after PR")
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(graph_dir, "results"))
    plt.close()

def before_after_graph(results_fn, graph_dir):
    errors_before = []
    errors_after = []
    with open(results_fn, "r") as f:
        results = json.load(f)
        num_seams = 0
        for seams_data in results.values():
            num_seams += len(seams_data.keys())
            for errors in seams_data.values():
                errors_before.append(errors[0])
                errors_after.append(errors[1])
        print("Number of parts: ", len(results.keys()))
        print("Total number of seams: ", num_seams)

    make_beforeAfter_plot(errors_before, errors_after, graph_dir)

def max_dist(pts1, pts2):
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    dists = []
    for p in pts1:
        d, ind = tree2.query([p], k=1)
        dists.append(d[0])

    for p in pts2:
        d, ind = tree1.query([p], k=1)
        dists.append(d[0])

    return np.max(dists)

def max_normal_dist(pts1, pts2):
    def normal_dist(vec1, vec2):
        vec1 = vec1.reshape(1,-1)
        vec2 = vec2.reshape(1,-1)
        n_vec1 = vec1 / np.linalg.norm(vec1)
        n_dist = np.cross(n_vec1, vec2)
        return np.linalg.norm(n_dist)

    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    dists = []
    for p in pts1:
        d, ind = tree2.query([p], k=2)
        vec1 = pts2[ind[0]] - pts2[ind[1]]
        vec2 = p - pts2[ind[0]]
        dists.append(normal_dist(vec1, vec2))

    for p in pts2:
        d, ind = tree1.query([p], k=2)
        vec1 = pts1[ind[0]] - pts1[ind[1]]
        vec2 = p - pts1[ind[0]]
        dists.append(normal_dist(vec1, vec2))

    return np.max(dists)

def get_input_data_list(data_dir):

    model_seam_scan_camera_fn_list = []
    parts_list = os.listdir(data_dir)
    
    for part in parts_list:
        part_dir = os.path.join(data_dir, part)
        model_mesh_fn = os.path.join(part_dir, "mesh.ply")
        
        seam_folder_list = os.listdir(part_dir)
        for seam_folder in seam_folder_list:
            if seam_folder == "mesh.ply":
                continue
            seam_folder_dir = os.path.join(part_dir, seam_folder)

            scan_pcd_fn = os.path.join(seam_folder_dir, "scan.ply")
            seam_fn = os.path.join(seam_folder_dir, "seam.ply")
            camera_config_fn = os.path.join(seam_folder_dir, "camera_config.yaml")
 
            model_seam_scan_camera_fn_list.append([part, seam_folder,model_mesh_fn, seam_fn, scan_pcd_fn, camera_config_fn])
    return model_seam_scan_camera_fn_list

def save_results(results_filename, results):
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=4)


def update_results(results_filename, template_name, key, error_before, error_after):
    if os.path.exists(results_filename):
        with open(results_filename, "r") as f:
            results = json.load(f)
    else:
        results = {}
    if template_name not in results:
        results[template_name] = {}
    results[template_name][key] = [error_before, error_after]
    save_results(results_filename, results)

def get_tranform_diff(transform1, transform2):
    print(transform1, transform2)
    rot1 = transform1[:3,:3]
    trans1 = transform1[:3,3]
    rot2 = transform2[:3,:3]
    trans2 = transform2[:3,3]

    translation_error = np.linalg.norm(trans1 - trans2)
    relative_rot = rot2 * np.linalg.inv(rot1)
    r = Rot.from_matrix(relative_rot)
    rot_vec = r.as_rotvec()
    rotation_error = np.linalg.norm(rot_vec)

    return translation_error, rotation_error


def run_partial_registration(transforms, executable, results_seam_dir, model_mesh_fn, seam_fn, scan_pcd_fn, camera_config_fn):
    seam = o3d.io.read_point_cloud(seam_fn)
    scan_pcd = o3d.io.read_point_cloud(scan_pcd_fn)
    model_mesh = o3d.io.read_triangle_mesh(model_mesh_fn)

    model_fn_transformed = "/tmp/model_pr.ply"
    seam_fn_transformed = "/tmp/seam_pr.ply"
    scan_pcd_noise_fn = "/tmp/scan_pr.ply"
    out_transform_filename = "/tmp/transform.txt"

    seam_center = seam.get_center()
    before_error_list = []
    after_error_list = []

    for trial_ind, transform in enumerate(transforms):
        # apply transformation wrt to the center of the scan
        np.savetxt(f"{results_seam_dir}/trail_{trial_ind}.txt", transform)
        temp_model_mesh = copy.deepcopy(model_mesh)
        temp_seam = copy.deepcopy(seam) 
        temp_scan = copy.deepcopy(scan_pcd)
        
        temp_scan_pcd_points = np.array(temp_scan.points)
        temp_scan.points = o3d.utility.Vector3dVector(temp_scan_pcd_points + np.random.normal(0.0, 0.001, np.shape(temp_scan_pcd_points)))
        o3d.io.write_point_cloud(scan_pcd_noise_fn, temp_scan)

        transform_center = np.eye(4)
        transform_center[:3,3] = seam_center
        transform_neg_center = np.eye(4)
        transform_neg_center[:3,3] = -seam_center
        origin_transform = transform_center @ transform @ transform_neg_center

        temp_model_mesh.transform(origin_transform)
        temp_seam.transform(origin_transform)
        temp_model_mesh.paint_uniform_color((0,1,0))

        error_before = max_dist(np.array(temp_seam.points), np.array(seam.points))

        o3d.io.write_triangle_mesh(model_fn_transformed, temp_model_mesh)
        o3d.io.write_point_cloud(seam_fn_transformed, temp_seam)

        # o3d.visualization.draw_geometries([ temp_model_mesh, temp_seam, scan_pcd])

        lib_so = "LD_LIBRARY_PATH=/home/amrishbaskaran/projects/GlobalSensorOptimization/build/install/lib/global_sensor_optimization/ "
        cmd = f"{lib_so} {executable} {model_fn_transformed} {scan_pcd_noise_fn} {camera_config_fn} {seam_fn_transformed}"
        print("\n", cmd)
        with open(os.devnull, 'wb') as shutup:
            subprocess.call(cmd, shell=True, stdout=shutup, stderr=shutup)
            # subprocess.call(cmd, shell=True)
        if os.path.exists(out_transform_filename):
            out_transform = np.loadtxt(out_transform_filename)
            np.savetxt(f"{results_seam_dir}/trail_{trial_ind}_pr.txt", out_transform)
            
            seam_pr = copy.deepcopy(temp_seam)
            seam_pr.transform(out_transform)

            error_after = max_dist(np.array(seam_pr.points), np.array(seam.points))
            os.remove(out_transform_filename) 
        else:
            error_after = -1.0
        after_error_list.append(error_after)
        before_error_list.append(error_before) 
        print(f"Trial {trial_ind} error- before: {error_before}, after:{error_after}")
    return before_error_list, after_error_list

def make_T(translation, rpy=None, rotVec=None):
    T = np.eye(4)
    T[:3, -1] = translation
    if rpy is not None:
        T[:3, :3] = Rot.from_euler("xyz", rpy).as_dcm()
    elif rotVec is not None:
        T[:3, :3] = Rot.from_rotvec(rotVec).as_matrix()
    return T

def gen_transforms(options):
    transforms = []
    mods = options["mods"]
    num_trials_per_mod = options["benchmark_settings"]["num_trials_per_mod"]
    for translation in mods["translation"]:
        for rot_angle in mods["rotation"]:
            # repeat for given number of trials
            for _ in range(num_trials_per_mod):
                random_direction = np.random.rand(3)
                random_dirvec = random_direction / np.linalg.norm(random_direction)
                random_rot_vec = random_dirvec * rot_angle
                
                random_direction = np.random.rand(3)
                random_dirvec = random_direction / np.linalg.norm(random_direction)
                random_translation = random_dirvec * translation
                
                transforms.append(make_T(random_translation, rotVec=random_rot_vec))
    return transforms

def benchmark_data(result_parent_dir, result_fn, executable, data_dir, num_trials_per_mod = 2):
    
    part_model_seam_ref_refseam_fn_list = get_input_data_list(data_dir)
    
    options = {}
    options["benchmark_settings"] = {"num_trials_per_mod": num_trials_per_mod}
    resolution = 5
    mods = {
        "translation": np.linspace(0, 0.08, resolution),
        "rotation": np.linspace(0, np.deg2rad(5), resolution),
    }
    options["mods"] = {k: v.tolist() for k, v in mods.items()}

    for part_name, seam_name, model_mesh_fn, seam_fn, scan_pcd_fn, camera_config_fn in part_model_seam_ref_refseam_fn_list:
        result_sub_dir = os.path.join(result_parent_dir,part_name)
        results_seam_dir = os.path.join(result_sub_dir,seam_name)
        if not os.path.exists(result_sub_dir):
            os.mkdir(result_sub_dir)
        if not os.path.exists(results_seam_dir):
            os.mkdir(results_seam_dir)
        
        # create transforms/ distortions
        transforms = gen_transforms(options)
        error_before_list, error_after_list = run_partial_registration(transforms, executable, results_seam_dir, model_mesh_fn, seam_fn, scan_pcd_fn, camera_config_fn)
        
        update_results(result_fn, part_name, "_" + seam_name, error_before_list, error_after_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-d",
        "--data_dirs",
        type=str,
        required=True,
        help="directories to run partial registration on",
        dest="data_dirs",
    )
    parser.add_argument(
        "-e",
        "--executable",
        type=str,
        required=True,
        help="path to partial registration executable",
        default=None,
        dest="executable",
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        type=str,
        required=True,
        help="path to dir for results",
        dest="results_dir",
    )
    parser.add_argument(
        "-p",
        "--postfix",
        type=str,
        required=False,
        help="postfix to add to results dir name (commit hash, maybe?)",
        default="mainBranch",
        dest="postfix",
    )

    args = parser.parse_args()
    results_dir = args.results_dir
    postfix = args.postfix

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_folder = os.path.join(results_dir, "{}_{}".format(timestamp, postfix))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    executable = args.executable
    data_dir = args.data_dirs

    result_parent_dir = os.path.join(output_folder,data_dir.split("/")[-1])
    
    if not os.path.exists(result_parent_dir):
        os.mkdir(result_parent_dir)

    result_fn = os.path.join(result_parent_dir, "results.json")

    start_time = time()
    benchmark_data(result_parent_dir, result_fn, executable, data_dir)

    end_time = time()

    print("Benchmark runtime: ", end_time - start_time)

    # folder = "/home/amrishbaskaran/benchmark_data/partial_registration/results/2022_03_08_10_24_33_test_normals/parts"
    # folder = "/home/amrishbaskaran/benchmark_data/partial_registration/results/2022_03_07_17_58_03_test_no_normals/parts"
    # result_fn = f"{folder}/results.json"
    # result_parent_dir = folder
    before_after_graph(result_fn, result_parent_dir)