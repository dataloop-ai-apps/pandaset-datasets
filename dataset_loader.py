import os.path
import shutil
import dtlpy as dl
import numpy as np
import open3d as o3d
import glob
from pandaset_devkit.python import pandaset
import json
from io import BytesIO
from dtlpylidar.parsers import pandaset_parser
from dtlpylidar.utilities.transformations import calc_transform_matrix, rotation_matrix_from_quaternion
from scipy.spatial.transform import Rotation as R
import urllib.request
import urllib.error
import zipfile


class PandasetLoader(pandaset_parser.PandaSetParser):
    def __init__(self):
        super().__init__()
        self.dataset_url = "https://storage.googleapis.com/model-mgmt-snapshots/datasets-lidar-pandaset/098.zip"
        self.ids_mapping = dict()

    @staticmethod
    def copy_calibration_files(source, dist):
        shutil.copy(source, dist)

    def get_object_id(self, uuid):
        if uuid not in self.ids_mapping:
            self.ids_mapping[uuid] = len(self.ids_mapping)
        return self.ids_mapping[uuid]

    @staticmethod
    def upload_sem_ref_items(frames_item: dl.Item, ref_items_dict, builder, semseg_classes):
        annotations_list = list()
        dataset = frames_item.dataset
        for annotation_uid, annotation_data in ref_items_dict.items():
            buffer = BytesIO()
            buffer.write(json.dumps(annotation_data.get('ref_item_json')).encode())
            buffer.seek(0)
            buffer.name = f"{annotation_uid}.json"
            sem_ref_item = dataset.items.upload(
                remote_path="/.dataloop/sem_ref",
                local_path=buffer,
                overwrite=True
            )
            ann_def = {"type": "ref_semantic_3d",
                       "label": semseg_classes.get(annotation_uid, 'Unknown'),
                       "coordinates": {
                           "interpolation": "none",
                           "mode": "overwrite",
                           "ref": f"{sem_ref_item.id}",
                           "refType": "id"
                       },
                       "metadata": {
                           "system": {
                               "frame": int(min(annotation_data.get('ref_item_json').get('frames').keys())),
                               "endFrame": int(max(annotation_data.get('ref_item_json').get('frames').keys()))
                           }
                       }}
            annotations_list.append(ann_def)
        return annotations_list

    @staticmethod
    def calculate_cube_corners(center, dimensions):
        """
        Calculates the 3D coordinates of all eight corners of a cube given its center and dimensions.

        Args:
            center: A list or numpy array of size 3 representing the center point (x, y, z).
            dimensions: A list or numpy array of size 3 representing the dimensions (x, y, z) of the cube.

        Returns:
            A numpy array of size (8, 3) containing the coordinates of all eight corners.
        """

        # Half dimensions for easier calculations
        half_dimensions = np.array(dimensions) / 2

        # Create a list of corner offsets relative to the center
        corner_offsets = [
            [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1]
        ]

        # Convert corner offsets to numpy array
        corner_offsets = np.array(corner_offsets)

        # Calculate corner positions by adding offsets to center and multiplying by half dimensions
        corners = center + corner_offsets * half_dimensions

        return corners

    def upload_annotations(self, item: dl.Item, sequence, max_frames=5):
        builder = item.annotations.builder()
        for frame_idx, cuboids in enumerate(sequence.cuboids.data):
            if frame_idx >= max_frames:
                break
            _rotation = sequence.lidar.poses[frame_idx].get('heading')
            _translation = sequence.lidar.poses[frame_idx].get('position')
            rotation = [_rotation.get('x', 0),
                        _rotation.get('y', 0),
                        _rotation.get('z', 0),
                        _rotation.get('w', 0)]

            translation = np.array([
                _translation.get('x', 0),
                _translation.get('y', 0),
                _translation.get('z', 0)
            ])
            rotation_matrix = rotation_matrix_from_quaternion(quaternion_x=rotation[0],
                                                              quaternion_y=rotation[1],
                                                              quaternion_z=rotation[2],
                                                              quaternion_w=rotation[3])
            transformation_matrix = calc_transform_matrix(rotation=rotation_matrix,
                                                          position=translation)
            for cube_idx, row in cuboids.iterrows():
                yaw = row['yaw']
                center = [row['position.x'], row['position.y'], row['position.z']]
                dimensions = [row['dimensions.x'], row['dimensions.y'], row['dimensions.z']]
                corners = self.calculate_cube_corners(center, dimensions)
                v3d = o3d.utility.Vector3dVector(corners)
                cloud = o3d.geometry.PointCloud(v3d)
                cloud.transform(transformation_matrix)
                box_rotation = R.from_euler('xyz', [0, 0, float(yaw)]).as_matrix()
                new_box_rotation = list(
                    R.from_matrix(np.dot(transformation_matrix[:3, :3], box_rotation)).as_euler('xyz'))

                builder.add(annotation_definition=dl.Cube3d(label=row['label'],
                                                            position=cloud.get_center() - translation,
                                                            rotation=new_box_rotation,
                                                            scale=list(dimensions),
                                                            ),
                            frame_num=frame_idx,
                            end_frame_num=frame_idx,
                            object_id=self.get_object_id(row['uuid']))
        semseg_classes = sequence.semseg.classes
        semantics_mapping = dict()
        for frame_idx, semantic_segmentations in enumerate(sequence.semseg.data):
            if frame_idx >= max_frames:
                break
            for seg_idx, row in semantic_segmentations.iterrows():
                _class = str(row['class'])
                if _class not in semantics_mapping:
                    semantics_mapping[_class] = {
                        "ref_item_json": {
                            "type": "index",
                            "frames": {}
                        },
                        "label": semseg_classes[_class]
                    }
                if str(frame_idx) not in semantics_mapping[_class]['ref_item_json']['frames']:
                    semantics_mapping[_class]['ref_item_json']['frames'][str(frame_idx)] = list()
                semantics_mapping[_class]['ref_item_json']['frames'][str(frame_idx)].append(seg_idx)
        semantic_annotations = self.upload_sem_ref_items(frames_item=item,
                                                         ref_items_dict=semantics_mapping,
                                                         builder=builder,
                                                         semseg_classes=semseg_classes)
        item.annotations.upload(builder)
        item.annotations.upload(semantic_annotations)
        print('1')

    def upload_data(self, dataset: dl.Dataset, path, sequence_name='098', max_frames=5):

        # Load Sequence
        pandaset_dataset = pandaset.DataSet(path)
        sequence = pandaset_dataset[sequence_name]
        sequence.load()

        # Sequence Directory
        sequence_dir_path = os.path.join(path, f"{sequence_name}")

        # Data directory
        write_path = os.path.join(os.getcwd(), f"{sequence_name}")
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        os.makedirs(write_path, exist_ok=True)

        pcds_path = os.path.join(write_path, "velodyne_points")
        os.makedirs(pcds_path, exist_ok=True)

        # Copy lidar calibration files

        # json_files = glob.glob(f"{sequence_dir_path}/lidar/*.json")
        json_files = glob.glob(os.path.join(sequence_dir_path, 'lidar', '*.json'))
        for json_file in json_files:
            self.copy_calibration_files(source=json_file,
                                        dist=os.path.join(pcds_path, os.path.basename(json_file)))

        for i in range(max_frames):
            data = sequence.lidar.data[i]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(data)[:, :3])
            pcd_path = os.path.join(pcds_path, f"{i}.pcd")
            o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=True)

        for camera_name, camera_data in sequence.camera.items():
            camera_path = os.path.join(write_path, camera_name)
            os.makedirs(camera_path, exist_ok=True)
            # json_files = glob.glob(f"{sequence_dir_path}/camera/{camera_name}/*.json")
            json_files = glob.glob(os.path.join(sequence_dir_path, 'camera', camera_name, '*.json'))
            for json_file in json_files:
                self.copy_calibration_files(source=json_file,
                                            dist=os.path.join(camera_path, os.path.basename(json_file)))

            for idx, img in enumerate(camera_data.data):
                if idx >= max_frames:
                    break
                img_path = os.path.join(camera_path, f"{idx}.jpg")
                img.save(img_path)

        # Upload data
        dataset.items.upload(local_path=write_path)
        scene_dir = dataset.items.get(filepath=f"/{sequence_name}", is_dir=True)
        item_id = self.create_json_calibration_files(scene_dir=scene_dir)
        scene_item = dl.items.get(item_id=item_id)
        self.upload_annotations(item=scene_item, sequence=sequence, max_frames=max_frames)

    def upload_dataset(self, dataset: dl.Dataset, source: str):
        path = os.path.join(os.getcwd(), 'data')
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, '098.zip')
        try:
            urllib.request.urlretrieve(source, zip_path)
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"Error downloading data: {e}")

        if not os.path.isfile(zip_path) or not zip_path.endswith(".zip"):
            raise FileNotFoundError(f"Error: '{zip_path}' is not a valid zip file.")

        # Extract zip contents to a directory with the same name (excluding .zip)
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(os.path.dirname(zip_path))
        zip_ref.close()

        print(f"Extracted contents of '{zip_path}' to the same directory.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        self.upload_data(dataset=dataset, path=path, sequence_name='098', max_frames=5)


if __name__ == '__main__':

    try:
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/model-mgmt-snapshots/datasets-lidar-pandaset/098.zip',
            '098.zip')
    except urllib.error.URLError as e:
        print(f"Error downloading data: {e}")
        exit()
    # project = dl.projects.get(project_name="ShadiProject")
    # # dataset = project.datasets.create(dataset_name="Pandaset Annotations")
    # dataset = project.datasets.get('Pandaset New 2')
    # parser = NewParser()
    # path = r'C:\Users\Shadi\Desktop\DL-Apps\dtlpy-lidar\New folder'
    # sequence_name = '098'
    # parser.upload_data(dataset,
    #                    path,
    #                    sequence_name=sequence_name,
    #                    max_frames=5)
    # dataset.open_in_web()
    # # pandaset_dataset = pandaset.DataSet(path)
    # # sequence = pandaset_dataset[sequence_name]
    # # sequence.load()
    # #
    # # item = dl.items.get(item_id='6686936a881a116e8417ef10')
    # parser.upload_annotations(item=item, sequence=sequence, max_frames=5)
