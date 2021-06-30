import os
import supervisely_lib as sly
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
import pandas as pd
import copy, json
from operator import add
from collections import defaultdict

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ["modal.state.slyProjectId"])
TASK_ID = int(os.environ["TASK_ID"])
OBJECTS = '_objects'
FIGURES = '_figures'
FRAMES = '_frames'
CLASS_NAME = 'class_name'
CLASSES = 'Classes'
TAGS = 'Tags'

TOTAL = 'TOTAL NUMBER OF TAGS'
TOTAL_COL = 'TOTAL'
PROJECT_COL = 'PROJECT'
COUNT_SUFFIX = '_cnt'
TAG_COLOMN = 'tag'
TAG_VALUE_COLOMN = 'tag_value'
FIRST_STRING = '#'


def process_images_tags_1(curr_image_tags, ds_images_tags_1):
    for tag in curr_image_tags:
        ds_images_tags_1[tag.name] += 1


def get_pd_tag_stat_1(meta, datasets, columns):
    data = []
    for idx, tag_meta in enumerate(meta.tag_metas):
        name = tag_meta.name
        row = [idx, name]
        row.extend([0])
        for ds_name, ds_property_tags in datasets:
            row.extend([ds_property_tags[name]])
            row[2] += ds_property_tags[name]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    total_row = list(df.sum(axis=0))
    total_row[0] = len(df)
    total_row[1] = TOTAL
    df.loc[len(df)] = total_row

    return df


def process_images_tags_2(curr_image_tags, image_info, ds_tags_to_imgs_urls):
    for tag in curr_image_tags:
        ds_tags_to_imgs_urls[tag.name].append('<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'
                         .format(image_info.full_storage_url, image_info.name))


def get_pd_tag_stat_2(meta, datasets, columns):
    data = []
    idx = 0
    for tag_meta in meta.tag_metas:
        name = tag_meta.name

        all_ds_rows = ['' for _ in datasets]

        for ds_index, (ds_name, ds_property_tags) in enumerate(datasets):
            for curr_link in ds_property_tags[name]:
                row = [idx, name]
                row.append(curr_link)
                row.extend(all_ds_rows)
                row[3 + ds_index] = curr_link
                idx += 1
                data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


def process_images_tags_3(curr_image_tags, ds_images_tags_vals_3, tags_to_vals):
    for tag in curr_image_tags:
        if tag.value not in tags_to_vals[tag.name]:
            tags_to_vals[tag.name].append(tag.value)
        ds_images_tags_vals_3[tag.name][tag.value] += 1


def get_pd_tag_stat_3(datasets, columns, tags_to_vals):

    data = []

    for idx, tag_name in enumerate(tags_to_vals):
        for tag_val in tags_to_vals[tag_name]:
            row = [idx, tag_name, tag_val]
            row.extend([0])
            for ds_name, ds_property_tags in datasets:
                row.extend([ds_property_tags[tag_name][tag_val]])
                row[3] += ds_property_tags[tag_name][tag_val]
            data.append(row)

    df = pd.DataFrame(data, columns=columns)
    total_row = list(df.sum(axis=0))
    total_row[0] = len(df)
    total_row[1] = TOTAL
    total_row.insert(2, '')
    df.loc[len(df)] = total_row

    return df


def process_images_tags_4(curr_image_tags, image_info, ds_tags_to_imgs_urls_4):
    for tag in curr_image_tags:
        ds_tags_to_imgs_urls_4[tag.name][tag.value].append('<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'
                                              .format(image_info.full_storage_url, image_info.name))


def get_pd_tag_stat_4(datasets, columns, tags_to_vals):
    data = []
    idx = 0
    for tag_name in tags_to_vals:
        for tag_val in tags_to_vals[tag_name]:

            all_ds_rows = ['' for _ in datasets]

            for ds_index, (ds_name, ds_property_tags) in enumerate(datasets):
                for curr_link in ds_property_tags[tag_name][tag_val]:
                    row = [idx, tag_name, tag_val]
                    row.append(curr_link)
                    row.extend(all_ds_rows)
                    row[4 + ds_index] = curr_link
                    idx += 1
                    data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


@my_app.callback("images_tags_stats")
@sly.timeit
def images_tags_stats(api: sly.Api, task_id, context, state, app_logger):

    project_info = api.project.get_info_by_id(PROJECT_ID)
    meta_json = api.project.get_meta(project_info.id)
    meta = sly.ProjectMeta.from_json(meta_json)

    if len(meta.tag_metas) == 0:
        app_logger.warn("Project {!r} have no tags".format(project_info.name))
        my_app.stop()

    columns_images_tags_1 = [FIRST_STRING, TAG_COLOMN, TOTAL_COL]
    datasets_counts_1 = []
    #========================================================================================== 2 ====
    columns_images_tags_2 = [FIRST_STRING, TAG_COLOMN, PROJECT_COL]
    datasets_counts_2 = []

    # ========================================================================================== 3 ====
    columns_images_tags_3 = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN, TOTAL_COL]
    datasets_counts_3 = []
    tags_to_vals = defaultdict(list)
    # =========================================================================================== 4 ====
    columns_images_tags_vals_4 = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN, PROJECT_COL]
    datasets_counts_4 = []



    id_to_tagmeta = meta.tag_metas.get_id_mapping()

    for dataset in api.dataset.get_list(PROJECT_ID):
        columns_images_tags_1.extend([dataset.name])                            # 1
        ds_images_tags_1 = defaultdict(int)                                     # 1

        columns_images_tags_2.extend([dataset.name])                            # 2
        ds_tags_to_imgs_urls_2 = defaultdict(list)                              # 2

        columns_images_tags_3.extend([dataset.name])                            # 3
        ds_images_tags_vals_3 = defaultdict(lambda: defaultdict(int))           # 3

        columns_images_tags_vals_4.extend([dataset.name])                       # 4
        ds_tags_to_imgs_urls_4 = defaultdict(lambda: defaultdict(list))         # 4

        images = api.image.get_list(dataset.id)

        for batch in sly.batched(images, batch_size=10):
            for image_info in batch:
                curr_image_tags = sly.TagCollection.from_api_response(image_info.tags, meta.tag_metas, id_to_tagmeta)
                process_images_tags_1(curr_image_tags, ds_images_tags_1)                            # 1
                #url = api.image.url(TEAM_ID, WORKSPACE_ID, PROJECT_ID, dataset.id, image_info.id)
                process_images_tags_2(curr_image_tags, image_info, ds_tags_to_imgs_urls_2)          # 2

                process_images_tags_3(curr_image_tags, ds_images_tags_vals_3, tags_to_vals)         # 3

                process_images_tags_4(curr_image_tags, image_info, ds_tags_to_imgs_urls_4)          # 4


        datasets_counts_1.append((dataset.name, ds_images_tags_1))                       # 1
        datasets_counts_2.append((dataset.name, ds_tags_to_imgs_urls_2))                 # 2
        datasets_counts_3.append((dataset.name, ds_images_tags_vals_3))                  # 3
        datasets_counts_4.append((dataset.name, ds_tags_to_imgs_urls_4))                 # 4

    df_1 = get_pd_tag_stat_1(meta, datasets_counts_1, columns_images_tags_1)                       # 1
    print(df_1)                                                                                    # 1

    df_2 = get_pd_tag_stat_2(meta, datasets_counts_2, columns_images_tags_2)                       # 2
    print(df_2)                                                                                    # 2

    df_3 = get_pd_tag_stat_3(datasets_counts_3, columns_images_tags_3, tags_to_vals)               # 3
    print(df_3)                                                                                    # 3

    df_4 = get_pd_tag_stat_4(datasets_counts_4, columns_images_tags_vals_4, tags_to_vals)          # 4
    print(df_4)                                                                                    # 4


    report_name = "{}_{}.lnk".format(PROJECT_ID, project_info.name)
    local_path = os.path.join(my_app.data_dir, report_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(my_app.app_url, file=text_file)
    remote_path = "/reports/images_tags_stat/{}".format(report_name)
    remote_path = api.file.get_free_name(TEAM_ID, remote_path)
    report_name = sly.fs.get_file_name_with_ext(remote_path)
    file_info = api.file.upload(TEAM_ID, local_path, remote_path)
    report_url = api.file.get_url(file_info.id)

    fields = [
        {"field": "data.loading", "payload": False},
        {"field": "data.imgs_tags_statTable", "payload": json.loads(df_1.to_json(orient="split"))},
        {"field": "data.tags_to_imgs_urls_statTable", "payload": json.loads(df_2.to_json(orient="split"))},
        {"field": "data.imgs_tags_vals_statTable", "payload": json.loads(df_3.to_json(orient="split"))},
        {"field": "data.tags_vals_to_imgs_urls_statTable", "payload": json.loads(df_4.to_json(orient="split"))},
        {"field": "data.savePath", "payload": remote_path},
        {"field": "data.reportName", "payload": report_name},
        {"field": "data.reportUrl", "payload": report_url},
    ]

    api.task.set_fields(task_id, fields)
    api.task.set_output_report(task_id, file_info.id, report_name)
    my_app.stop()

    a = 0




    if CLASSES in stat_type:
        classes = []
        counter = {}
        classes_id = []
        for idx, curr_class in enumerate(meta.obj_classes):
            classes.append(curr_class.name)
            classes_id.append(idx)
            counter[curr_class.name] = 0

        columns_classes = [FIRST_STRING, CLASS_NAME, 'total_objects', 'total_figures', 'total_frames']
        data = {FIRST_STRING: classes_id, CLASS_NAME: classes, 'total_objects': [0] * len(classes), 'total_figures': [0] * len(classes), 'total_frames': [0] * len(classes)}

    if TAGS in stat_type:
        columns = [FIRST_STRING, TAG_COLOMN]
        columns_for_values = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN]
        columns_frame_tag = [FIRST_STRING, TAG_COLOMN]  # ===========frame_tags=======
        columns_frame_tag_values = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN]  # ===========frame_tags=======
        columns_object_tag = [FIRST_STRING, TAG_COLOMN]  # ===========object_tags=======
        columns_object_tag_values = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN]  # ===========object_tags=======

        columns.extend([TOTAL])
        columns_for_values.extend([TOTAL])
        columns_frame_tag.extend([TOTAL, TOTAL + COUNT_SUFFIX])  # ===========frame_tags=======
        columns_frame_tag_values.extend([TOTAL])  # ===========frame_tags=======
        columns_object_tag.extend([TOTAL])  # ===========object_tags=======
        columns_object_tag_values.extend([TOTAL])  # ===========object_tags=======

        datasets_counts = []
        datasets_values_counts = []
        datasets_frame_tag_counts = []  # ===========frame_tags=======
        datasets_frame_tag_values_counts = []  # ===========frame_tags=======
        datasets_object_tag_counts = []  # ===========object_tags=======
        datasets_object_tag_values_counts = []  # ===========object_tags=======

    for dataset in api.dataset.get_list(PROJECT_ID):

        if CLASSES in stat_type:
            columns_classes.extend([dataset.name + OBJECTS, dataset.name + FIGURES, dataset.name + FRAMES])
            classes_counter = copy.deepcopy(counter)
            figures_counter = copy.deepcopy(counter)
            frames_counter = copy.deepcopy(counter)
            data[dataset.name + OBJECTS] = []
            data[dataset.name + FIGURES] = []
            data[dataset.name + FRAMES] = []
            videos = api.video.get_list(dataset.id)
            progress_classes = sly.Progress("Processing video classes ...", len(videos), app_logger)

        if TAGS in stat_type:
            columns.extend([dataset.name])
            ds_property_tags = defaultdict(int)

            columns_for_values.extend([dataset.name])
            ds_property_tags_values = defaultdict(lambda: defaultdict(int))

            # ===========frame_tags=========================================
            columns_frame_tag.extend([dataset.name, dataset.name + COUNT_SUFFIX])
            ds_frame_tags = defaultdict(int)
            ds_frame_tags_counter = defaultdict(int)

            columns_frame_tag_values.extend([dataset.name])
            ds_frame_tags_values = defaultdict(lambda: defaultdict(int))
            ds_frame_tags_values_counter = defaultdict(lambda: defaultdict(int))
            # ===========frame_tags=========================================

            # ===========object_tags=========================================
            columns_object_tag.extend([dataset.name])
            ds_object_tags = defaultdict(int)

            columns_object_tag_values.extend([dataset.name])
            ds_object_tags_values = defaultdict(lambda: defaultdict(int))
            # ===========object_tags=========================================

            videos = api.video.get_list(dataset.id)
            progress_tags = sly.Progress("Processing video tags ...", len(videos), app_logger)

        for batch in sly.batched(videos, batch_size=10):
            for video_info in batch:

                ann_info = api.video.annotation.download(video_info.id)
                ann = sly.VideoAnnotation.from_json(ann_info, meta, key_id_map)

                if CLASSES in stat_type:
                    classes_counter, figures_counter, frames_counter = items_counter(ann, classes_counter, figures_counter, frames_counter)
                    progress_classes.iter_done_report()

                if TAGS in stat_type:
                    process_video_annotation(ann, ds_property_tags)
                    process_video_annotation_tags_values(ann, ds_property_tags_values)

                    process_video_ann_frame_tags(ann, ds_frame_tags,
                                                 ds_frame_tags_counter)  # ===========frame_tags=======
                    process_video_ann_frame_tags_vals(ann, ds_frame_tags_values)  # ===========frame_tags=======

                    process_video_ann_object_tags(ann, ds_object_tags)  # ===========object_tags=======
                    process_video_ann_object_tags_vals(ann, ds_object_tags_values)  # ===========object_tags=======

                    progress_tags.iter_done_report()

        if CLASSES in stat_type:
            data = data_counter(data, dataset, classes, classes_counter, figures_counter, frames_counter)

        if TAGS in stat_type:
            datasets_counts.append((dataset.name, ds_property_tags))
            datasets_values_counts.append((dataset.name, ds_property_tags_values))
            datasets_frame_tag_counts.append((dataset.name, ds_frame_tags))  # ===========frame_tags=======
            datasets_frame_tag_values_counts.append(
                (dataset.name, ds_frame_tags_values))  # ===========frame_tags=======
            datasets_object_tag_counts.append((dataset.name, ds_object_tags))  # ===========object_tags=======
            datasets_object_tag_values_counts.append(
                (dataset.name, ds_object_tags_values))  # ===========object_tags=======

    if CLASSES in stat_type:
        classes.append(TOTAL)
        data[FIRST_STRING].append(len(data[FIRST_STRING]))
        for key, val in data.items():
            if key == CLASS_NAME or key == FIRST_STRING:
                continue
            data[key].append(sum(val))
        df_classes = pd.DataFrame(data, columns=columns_classes, index=classes)
        print(df_classes)

    if TAGS in stat_type:
        # =========property_tags===============================================================
        df = get_pd_tag_stat(meta, datasets_counts, columns)
        print('Total video tags stats')
        print(df)
        # =========property_tags_values=========================================================
        df_values = get_pd_tag_values_stat(datasets_values_counts, columns_for_values)
        print('Total video tags values stats')
        print(df_values)

        # =========frame_tag=====================================================================
        data_frame_tags = []
        for idx, tag_meta in enumerate(meta.tag_metas):
            name = tag_meta.name
            row_frame_tags = [idx, name]
            row_frame_tags.extend([0, 0])
            for ds_name, ds_frame_tags in datasets_frame_tag_counts:
                row_frame_tags.extend([ds_frame_tags[name], ds_frame_tags_counter[name]])
                row_frame_tags[2] += ds_frame_tags[name]
                row_frame_tags[3] += ds_frame_tags_counter[name]
            data_frame_tags.append(row_frame_tags)

        df_frame_tags = pd.DataFrame(data_frame_tags, columns=columns_frame_tag)
        total_row = list(df_frame_tags.sum(axis=0))
        total_row[0] = len(df_frame_tags)
        total_row[1] = TOTAL
        df_frame_tags.loc[len(df_frame_tags)] = total_row
        print('Total frame tags stats')
        print(df_frame_tags)

        # =========frame_tags_values=============================================================
        df_frame_tags_values = get_pd_tag_values_stat(datasets_frame_tag_values_counts, columns_frame_tag_values)
        print('Total frame tags values stats')
        print(df_frame_tags_values)

        # ==========object_tag================================================================
        df_object_tags = get_pd_tag_stat(meta, datasets_object_tag_counts, columns_object_tag)
        print('Total object tags stats')
        print(df_object_tags)
        # =========object_tags_values=========================================================
        df_object_values = get_pd_tag_values_stat(datasets_object_tag_values_counts, columns_object_tag_values)
        print('Total object tags values stats')
        print(df_object_values)

    report_name = "{}_{}.lnk".format(PROJECT_ID, project_info.name)
    local_path = os.path.join(my_app.data_dir, report_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(my_app.app_url, file=text_file)
    remote_path = "/reports/video_stat/{}".format(report_name)
    remote_path = api.file.get_free_name(TEAM_ID, remote_path)
    report_name = sly.fs.get_file_name_with_ext(remote_path)
    file_info = api.file.upload(TEAM_ID, local_path, remote_path)
    report_url = api.file.get_url(file_info.id)

    fields = [
        {"field": "data.loading", "payload": False},
        {"field": "data.classesTable", "payload": json.loads(df_classes.to_json(orient="split"))},
        {"field": "data.tagsTable", "payload": json.loads(df.to_json(orient="split"))},
        {"field": "data.savePath", "payload": remote_path},
        {"field": "data.reportName", "payload": report_name},
        {"field": "data.reportUrl", "payload": report_url},
    ]

    api.task.set_fields(task_id, fields)
    api.task.set_output_report(task_id, file_info.id, report_name)
    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": TEAM_ID,
        "WORKSPACE_ID": WORKSPACE_ID,
        "PROJECT_ID": PROJECT_ID
    })

    data = {
        "userImageTable": {"columns": [], "data": []}
    }

    my_app.run(data=data, initial_events=[{"command": "images_tags_stats"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
