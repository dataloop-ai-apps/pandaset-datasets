import os.path
import pathlib
import dtlpy as dl
import json
import urllib.request
import urllib.error
import zipfile


class PandasetLoader(dl.BaseServiceRunner):
    def __init__(self):
        super().__init__()
        # self.dataset_url = "https://storage.googleapis.com/model-mgmt-snapshots/datasets-lidar-pandaset/098.zip"
        self.dataset_url = "https://storage.googleapis.com/model-mgmt-snapshots/datasets-lidar-pandaset/001.zip"
        self.ontology_filename = "ontology.json"

    @staticmethod
    def upload_data(dataset: dl.Dataset, path, sequence_name="001"):
        # Upload scene folder
        scene_path = os.path.join(path, sequence_name)
        dataset.items.upload(local_path=scene_path)
        frames_item = dataset.items.get(filepath=f"/{sequence_name}/frames.json")
        frames_item_json = json.load(fp=frames_item.download(save_locally=False))

        # Update frames item
        download_path = dataset.download_annotations(
            local_path=path,
            annotation_options=dl.ViewAnnotationOptions.JSON
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

        # Upload annotations
        annotations_filepath = os.path.join(path, f'{sequence_name}_frames.json')
        builder = dl.AnnotationCollection.from_json_file(filepath=annotations_filepath)
        builder.item = frames_item
        builder.upload()
        return frames_item

    def _import_recipe_ontology(self, dataset: dl.Dataset):
        recipe = dataset.recipes.list()[0]
        ontology = recipe.ontologies.list()[0]

        new_ontology_filepath = os.path.join(os.path.dirname(str(__file__)), self.ontology_filename)
        with open(file=new_ontology_filepath, mode='r') as file:
            new_ontology_json = json.load(fp=file)

        new_ontology = dl.Ontology.from_json(_json=new_ontology_json, client_api=dl.client_api, recipe=recipe)
        new_ontology.id = ontology.id
        new_ontology.creator = ontology.creator
        new_ontology.metadata["system"]["projectIds"] = recipe.project_ids
        new_ontology.update()
        return recipe

    def upload_dataset(self, dataset: dl.Dataset, source: str):
        self._import_recipe_ontology(dataset=dataset)

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
        print(f"Extracted contents of '{zip_path}' to the same directory.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        self.upload_data(dataset=dataset, path=path)
