import os
import pathlib
import dtlpy as dl
import json
import urllib.request
import urllib.error
import zipfile
import logging

logger = logging.getLogger(name='pandaset-dataset')


class PandasetLoader(dl.BaseServiceRunner):
    def __init__(self):
        super().__init__()
        # self.dataset_url = "https://storage.googleapis.com/model-mgmt-snapshots/datasets-lidar-pandaset/098.zip"
        self.dataset_url = "https://storage.googleapis.com/model-mgmt-snapshots/datasets-lidar-pandaset/001.zip"
        self.ontology_filename = "ontology.json"

    def _import_recipe_ontology(self, dataset: dl.Dataset):
        recipe: dl.Recipe = dataset.recipes.list()[0]
        ontology: dl.Ontology = recipe.ontologies.list()[0]

        new_ontology_filepath = os.path.join(os.path.dirname(str(__file__)), self.ontology_filename)
        with open(file=new_ontology_filepath, mode='r') as file:
            new_ontology_json = json.load(fp=file)

        new_ontology = ontology.copy_from(ontology_json=new_ontology_json)
        return new_ontology

    @staticmethod
    def download_zip(source, progress=None):
        if progress is not None:
            progress.update(progress=10, message="Downloading dataset for source...")

        path = os.path.join(os.getcwd(), 'data')
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, '001.zip')
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
        logger.info(f"Extracted contents of '{zip_path}' to the same directory.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return path

    @staticmethod
    def _upload_data(dataset: dl.Dataset, path, sequence_name="001", progress=None):
        if progress is not None:
            progress.update(progress=40, message="Uploading source data...")

        # Add progress update callback
        progress_tracker = {'last_progress': 0}
        def progress_callback(**kwargs):
            p = kwargs.get('progress')  # p is between 0-100
            if progress is not None:
                progress_int = round(p / 10) * 10  # round to 10th
                if progress_int % 10 == 0 and progress_int != progress_tracker['last_progress']:
                    progress.update(progress=40 + (40 * progress_int / 100), message="Uploading source data...")
                    progress_tracker['last_progress'] = progress_int
        dl.client_api.callbacks.add(event='itemUpload', func=progress_callback)

        # Upload scene folder
        scene_path = os.path.join(path, sequence_name)
        dataset.items.upload(local_path=scene_path)
        frames_item = dataset.items.get(filepath=f"/{sequence_name}/frames.json")
        frames_item_json = json.load(fp=frames_item.download(save_locally=False))

        # Update frames item
        download_path = dataset.download_annotations(
            local_path=path,
            annotation_options=dl.ViewAnnotationOptions.JSON,
            overwrite=True
        )
        jsons_path = os.path.join(download_path, 'json')
        json_filepaths = pathlib.Path(jsons_path).rglob('*.json')
        for json_filepath in json_filepaths:
            json_filepath = str(json_filepath)
            with open(json_filepath, 'r') as fp:
                item_json = json.load(fp=fp)
            item = dl.Item.from_json(_json=item_json, client_api=dl.client_api)

            # PCD item
            if 'pcd' in item.mimetype:
                frame_number = int(item.name.rstrip('.pcd'))
                frames_item_json['frames'][frame_number]['lidar']['lidar_pcd_id'] = item.id
            # Camera item
            elif 'image' in item.mimetype:
                frame_number = int(item.name.rstrip('.jpg'))
                frame_images = frames_item_json['frames'][frame_number]['images']
                for idx, image in enumerate(frame_images):
                    if item.filename == image['remote_path']:
                        frames_item_json['frames'][frame_number]['images'][idx]['image_id'] = item.id
            else:
                continue

        # Upload frames item
        frames_item = dataset.items.upload(
            remote_name=frames_item.name,
            remote_path=frames_item.dir,
            local_path=json.dumps(frames_item_json).encode(),
            overwrite=True,
            item_metadata={
                "system": {
                    "shebang": {
                        "dltype": "PCDFrames"
                    }
                },
                "fps": 1
            }
        )
        return frames_item

    @staticmethod
    def _upload_annotations(frames_item: dl.Item, path, sequence_name="001", include_semantic=False, progress=None):
        if progress is not None:
            progress.update(progress=90, message="Uploading annotations...")

        # Load annotations and modify them
        annotations_filepath = os.path.join(path, f'{sequence_name}_frames.json')
        builder = dl.AnnotationCollection.from_json_file(filepath=annotations_filepath)

        # TODO: Segmentation TBD
        if include_semantic is True:
            # Get and Upload Segmentation References
            dataset = dl.datasets.get(dataset_id=frames_item.dataset.id)
            sem_ref_path = os.path.join(path, 'sem_ref')
            sem_ref_items = dataset.items.upload(local_path=sem_ref_path, remote_path="/.dataloop")
            sem_ref_items_map = {pathlib.Path(item.filename).stem.split('_', 1)[1]: item for item in sem_ref_items}

            annotation: dl.Annotation
            for annotation in builder.annotations:
                if annotation.type == "ref_semantic_3d":
                    ref_item = sem_ref_items_map[annotation.label]
                    annotation.coordinates["ref"] = ref_item.id
        else:
            # Remove all semantic annotations
            cubes_annotations = list()
            for annotation in builder.annotations:
                if annotation.type == dl.AnnotationType.CUBE3D:
                    cubes_annotations.append(annotation)
            builder.annotations = cubes_annotations

        # Upload annotations
        builder.item = frames_item
        builder.upload()

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress: dl.Progress = None):
        self._import_recipe_ontology(dataset=dataset)
        path = self.download_zip(source=source, progress=progress)
        frames_item = self._upload_data(dataset=dataset, path=path, progress=progress)
        self._upload_annotations(frames_item=frames_item, path=path, progress=progress)
        return frames_item
