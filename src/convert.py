import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    images_path = "/home/alex/DATASETS/TODO/PAXRAY++/paxray_images_unfiltered/images_patlas"

    batch_size = 10

    train_json = "/home/alex/DATASETS/TODO/PAXRAY++/paxray_train_val.json"
    test_json = "/home/alex/DATASETS/TODO/PAXRAY++/paxray_test.json"
    group_tag_name = "im id"

    def convert_rle_mask_to_polygon(rle_mask_data):
        if type(rle_mask_data["counts"]) is str:
            rle_mask_data["counts"] = bytes(rle_mask_data["counts"], encoding="utf-8")
            mask = mask_util.decode(rle_mask_data)
        else:
            rle_obj = mask_util.frPyObjects(
                rle_mask_data,
                rle_mask_data["size"][0],
                rle_mask_data["size"][1],
            )
            mask = mask_util.decode(rle_obj)
        mask = np.array(mask, dtype=bool)
        return sly.Bitmap(mask).to_contours()

    def create_ann(image_path):
        labels = []
        tags = []

        im_id_value = get_file_name(image_path)[:-8]
        group_tag = sly.Tag(group_tag_meta, value=im_id_value)
        tags.append(group_tag)

        if len(get_file_name(image_path).split("frontal")) == 1:
            view = sly.Tag(lateral_meta)
        else:
            view = sly.Tag(frontal_meta)

        tags.append(view)

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = 512
        img_wight = 512

        ann_data = image_name_to_ann_data[get_file_name_with_ext(image_path)]
        for curr_ann_data in ann_data:
            category_id = curr_ann_data[0]
            obj_class = idx_to_obj_class[category_id]
            rle_mask_data = curr_ann_data[1]
            polygons = convert_rle_mask_to_polygon(rle_mask_data)
            for polygon in polygons:
                label = sly.Label(polygon, obj_class)
                labels.append(label)

            bbox_coord = curr_ann_data[2]
            rectangle = sly.Rectangle(
                top=int(bbox_coord[1]),
                left=int(bbox_coord[0]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
                right=int(bbox_coord[0] + bbox_coord[2]),
            )
            label_rectangle = sly.Label(rectangle, obj_class)
            labels.append(label_rectangle)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    frontal_meta = sly.TagMeta("frontal", sly.TagValueType.NONE)
    lateral_meta = sly.TagMeta("lateral", sly.TagValueType.NONE)
    group_tag_meta = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)

    meta = sly.ProjectMeta(tag_metas=[frontal_meta, lateral_meta, group_tag_meta])
    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)

    idx_to_obj_class = {}
    image_id_to_name = {}
    image_name_to_ann_data = defaultdict(list)

    # ds_name_to_anns = {"train": train_json, "test": test_json}
    ds_name_to_anns = {"test": test_json}

    for ds_name, ann_json in ds_name_to_anns.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
        ann = load_json_file(ann_json)
        for curr_category in ann["categories"]:
            if idx_to_obj_class.get(curr_category["id"]) is None:
                obj_class = sly.ObjClass(curr_category["name"].rstrip(), sly.AnyGeometry)
                meta = meta.add_obj_class(obj_class)
                idx_to_obj_class[curr_category["id"]] = obj_class
        api.project.update_meta(project.id, meta.to_json())

        for curr_image_info in ann["images"]:
            image_id_to_name[curr_image_info["id"]] = curr_image_info["file_name"]

        for curr_ann_data in ann["annotations"]:
            image_id = curr_ann_data["image_id"]
            image_name_to_ann_data[image_id_to_name[image_id]].append(
                [curr_ann_data["category_id"], curr_ann_data["segmentation"], curr_ann_data["bbox"]]
            )

        images_names = list(image_name_to_ann_data.keys())

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, image_path) for image_path in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
