import os
import supervisely_lib as sly
import pandas as pd
import json
from collections import defaultdict
from supervisely_lib.annotation.tag_meta import TagApplicableTo

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ["modal.state.slyProjectId"])
TASK_ID = int(os.environ["TASK_ID"])
CLASS_NAME = 'class_name'
CLASSES = 'Classes'
TAGS = 'Tags'

TOTAL = 'TOTAL NUMBER OF TAGS'
TOTAL_COL = 'TOTAL'
PROJECT_COL = 'PROJECT'
IMAGE_COL = 'IMAGE NAME'
DATASET_NAME = 'DATASET NAME'
NUM_OBJECTS = 'NUMBER OF OBJECTS'
OBJECT_CLASS = 'OBJECT CLASS'
COUNT_SUFFIX = '_cnt'
TAG_COLOMN = 'tag'
TAG_VALUE_COLOMN = 'tag_value'
FIRST_STRING = '#'
logger = my_app.logger
objects_tags = []

images_tags_to_values = defaultdict(list)
objects_tags_to_values = defaultdict(list)


def get_images_tags_vals(state_vals):

    curr_objs_vals = defaultdict(list)
    for item in state_vals:
        logger.warn('{}'.format(item))
        tag_val, tag_name, tag_type = item.split(' ')
        if tag_type == 'int':
            tag_val = int(tag_val)
        elif tag_type == 'float':
            tag_val = float(tag_val)
        curr_objs_vals[tag_name].append(tag_val)

    for tag, val in images_tags_to_values.items():
        if tag not in curr_objs_vals.keys():
            curr_objs_vals[tag].extend(val)

    return curr_objs_vals


def get_objects_tags_vals(state_vals):

    curr_objs_vals = defaultdict(list)
    for item in state_vals:
        tag_val, tag_name, tag_type = item.split(' ')
        if tag_type == 'int':
            tag_val = int(tag_val)
        elif tag_type == 'float':
            tag_val = float(tag_val)
        curr_objs_vals[tag_name].append(tag_val)

    for tag, val in objects_tags_to_values.items():
        if tag not in curr_objs_vals.keys():
            curr_objs_vals[tag].extend(val)

    return curr_objs_vals


def process_images_tags_1(curr_image_tags, ds_images_tags_1, state):
    for tag in curr_image_tags:
        if tag.name in state['choose_tags']:
            ds_images_tags_1[tag.name] += 1


def get_pd_tag_stat_1(datasets, columns, state):
    data = []
    for idx, name in enumerate(state['choose_tags']):
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


def process_images_tags_2(curr_image_tags, image_info, ds_tags_to_imgs_urls, state):
    for tag in curr_image_tags:
        if tag.name in state['choose_tags']:
            ds_tags_to_imgs_urls[tag.name].append('<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'
                             .format(image_info.full_storage_url, image_info.name))


def get_pd_tag_stat_2(datasets, columns, state):
    data = []
    idx = 0
    for name in state['choose_tags']:
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


def process_images_tags_3(curr_image_tags, ds_images_tags_vals_3, tags_to_vals, state):

    curr_objs_vals = get_images_tags_vals(state['choose_vals'])

    for tag in curr_image_tags:
        if tag.name in state['choose_tags']:
            if tag.value in curr_objs_vals[tag.name] or len(state['choose_vals']) == 0:
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
    if len(data) > 0:
        if len(total_row) != len(data[0]):
            total_row.insert(2, '')
        else:
            total_row[2] = ''
    else:
        total_row[2] = ''
    df.loc[len(df)] = total_row

    return df


def process_images_tags_4(curr_image_tags, image_info, ds_tags_to_imgs_urls_4, state):
    curr_objs_vals = get_images_tags_vals(state['choose_vals'])
    for tag in curr_image_tags:
        if tag.name in state['choose_tags']:
            if tag.value in curr_objs_vals[tag.name] or len(state['choose_vals']) == 0:
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


def get_objects_tags(ann):
    tags = []
    for label in ann.labels:
        for curr_tag in label.tags:
            tags.append(curr_tag)
    return tags


def process_objects_tags_5(curr_object_tags, ds_objects_tags_5, state):
    for tag in curr_object_tags:
        if tag.name in state['choose_objs_tags']:
            ds_objects_tags_5[tag.name] += 1


def get_pd_tag_stat_5(datasets, columns, state):
    data = []
    for idx, name in enumerate(state['choose_objs_tags']):
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


def process_objects_tags_6(curr_object_tags, image_info, ds_obj_tags_to_imgs_urls_6, state):
    link = '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(image_info.full_storage_url, image_info.name)
    for tag in curr_object_tags:
        if tag.name in state['choose_objs_tags']:
            ds_obj_tags_to_imgs_urls_6[tag.name][link] += 1


def get_pd_tag_stat_6(datasets, columns, state):
    data = []
    idx = 0
    for name in state['choose_objs_tags']:

        all_ds_rows = ['' for _ in datasets]

        for ds_index, (ds_name, ds_property_tags) in enumerate(datasets):
            for curr_link in ds_property_tags[name]:
                row = [idx, name]
                row.append(curr_link)
                row.extend(all_ds_rows)
                row[3 + ds_index] = ds_property_tags[name][curr_link]
                idx += 1
                data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


def process_objects_tags_7(curr_object_tags, ds_objects_tags_vals_7, obj_tags_to_vals, state):

    curr_objs_vals = get_objects_tags_vals(state['choose_objs_vals'])
    for tag in curr_object_tags:
        if tag.name in state['choose_objs_tags']:
            if tag.value in curr_objs_vals[tag.name] or len(state['choose_objs_vals']) == 0:
                if tag.value not in obj_tags_to_vals[tag.name]:
                    obj_tags_to_vals[tag.name].append(tag.value)
                ds_objects_tags_vals_7[tag.name][tag.value] += 1


def get_pd_tag_stat_7(datasets, columns, obj_tags_to_vals):
    data = []

    for idx, tag_name in enumerate(obj_tags_to_vals):
        for tag_val in obj_tags_to_vals[tag_name]:
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
    if len(data) > 0:
        if len(total_row) != len(data[0]):
            total_row.insert(2, '')
        else:
            total_row[2] = ''
    else:
        total_row[2] = ''
    df.loc[len(df)] = total_row

    return df


def process_objects_tags_8(curr_object_tags, image_info, ds_tags_to_imgs_urls_8, state):
    link = '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(image_info.full_storage_url, image_info.name)
    curr_objs_vals = get_objects_tags_vals(state['choose_objs_vals'])
    for tag in curr_object_tags:
        if tag.name in state['choose_objs_tags']:
            if tag.value in curr_objs_vals[tag.name] or len(state['choose_objs_vals']) == 0:
                ds_tags_to_imgs_urls_8[tag.name][tag.value][link] += 1


def get_pd_tag_stat_8(datasets, columns, obj_tags_to_vals):
    data = []
    idx = 0
    for tag_name in obj_tags_to_vals:
        for tag_val in obj_tags_to_vals[tag_name]:

            all_ds_rows = ['' for _ in datasets]

            for ds_index, (ds_name, ds_property_tags) in enumerate(datasets):
                for curr_link in ds_property_tags[tag_name][tag_val]:
                    row = [idx, tag_name, tag_val]
                    row.append(curr_link)
                    row.extend(all_ds_rows)
                    row[4 + ds_index] = ds_property_tags[tag_name][tag_val][curr_link]
                    idx += 1
                    data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


def process_images_urls_to_img_tags_9(curr_image_tags, image_info, imgs_urls_ro_img_tags_9):
    link = '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(image_info.full_storage_url, image_info.name)
    for tag in curr_image_tags:
        imgs_urls_ro_img_tags_9[link][tag.name].append(tag.value)


def get_pd_tag_stat_9(datasets, columns):
    data = []
    idx = 0
    for dataset_name, links_to_tag_val in datasets:
        for link in links_to_tag_val:
            for tag_name in links_to_tag_val[link]:
                for tag_val in links_to_tag_val[link][tag_name]:
                    row = [idx, link, dataset_name, tag_name, tag_val]
                    idx += 1
                    data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


def process_images_urls_to_obj_tags_10(curr_object_tags, image_info, imgs_urls_to_obj_tags_10):
    link = '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(image_info.full_storage_url,
                                                                                    image_info.name)
    for tag in curr_object_tags:
        imgs_urls_to_obj_tags_10[link][tag.name][tag.value] += 1


def get_pd_tag_stat_10(datasets, columns):
    data = []
    idx = 0
    for dataset_name, links_to_tag_val in datasets:
        for link in links_to_tag_val:
            for tag_name in links_to_tag_val[link]:
                for tag_val in links_to_tag_val[link][tag_name]:
                    row = [idx, link, dataset_name, tag_name, tag_val, links_to_tag_val[link][tag_name][tag_val]]
                    idx += 1
                    data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


def process_obj_tags_to_class_11(ann, obj_tags_to_class_11, state):
    for label in ann.labels:
        for tag in label.tags:
            if tag.name in state['choose_objs_tags']:
                obj_tags_to_class_11[tag.name][label.obj_class.name] += 1


def get_pd_tag_stat_11(meta, datasets, columns, state):
    data = []
    idx = 0
    all_ds_rows = ['' for _ in datasets]
    for tag_name in state['choose_objs_tags']:
        for obj_class in meta.obj_classes:
            row = [idx, tag_name, obj_class.name, 0]
            row.extend(all_ds_rows)

            for ds_index, (dataset_name, tags_to_classes) in enumerate(datasets):
                row[3] += tags_to_classes[tag_name][obj_class.name]
                row[4 + ds_index] = tags_to_classes[tag_name][obj_class.name]

            if row[3] == 0:
                continue
            idx += 1
            data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


def process_obj_tags_to_class_12(ann, obj_tags_to_class_12, state):
    curr_objs_vals = get_objects_tags_vals(state['choose_objs_vals'])
    for label in ann.labels:
        for tag in label.tags:
            if tag.name in state['choose_objs_tags']:
                if tag.value in curr_objs_vals[tag.name] or len(state['choose_objs_vals']) == 0:
                    obj_tags_to_class_12[tag.name][tag.value][label.obj_class.name] += 1


def get_pd_tag_stat_12(meta, datasets, columns, obj_tags_to_vals):
    data = []
    idx = 0
    all_ds_rows = ['' for _ in datasets]
    for tag_meta in meta.tag_metas:
        tag_name = tag_meta.name
        for tag_val in obj_tags_to_vals[tag_name]:
            for obj_class in meta.obj_classes:
                row = [idx, tag_name, tag_val, obj_class.name, 0]
                row.extend(all_ds_rows)
                for ds_index, (dataset_name, tags_to_classes) in enumerate(datasets):
                    row[4] += tags_to_classes[tag_name][tag_val][obj_class.name]
                    row[5 + ds_index] = tags_to_classes[tag_name][tag_val][obj_class.name]

                if row[4] == 0:
                    continue
                idx += 1
                data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


@my_app.callback("get_statistics")
@sly.timeit
def get_statistics(api: sly.Api, task_id, context, state, app_logger):

    project_info = api.project.get_info_by_id(PROJECT_ID)
    meta_json = api.project.get_meta(project_info.id)
    meta = sly.ProjectMeta.from_json(meta_json)

    # ========================================================================================== 1 ====
    columns_images_tags_1 = [FIRST_STRING, TAG_COLOMN, TOTAL_COL]
    datasets_counts_1 = []
    # ========================================================================================== 2 ====
    columns_images_tags_2 = [FIRST_STRING, TAG_COLOMN, PROJECT_COL]
    datasets_counts_2 = []
    # ========================================================================================== 3 ====
    columns_images_tags_3 = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN, TOTAL_COL]
    datasets_counts_3 = []
    tags_to_vals = defaultdict(list)
    # =========================================================================================== 4 ====
    columns_images_tags_vals_4 = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN, PROJECT_COL]
    datasets_counts_4 = []
    # =========================================================================================== 5 ====
    columns_objects_tags_5 = [FIRST_STRING, TAG_COLOMN, TOTAL_COL]
    datasets_counts_5 = []
    # =========================================================================================== 6 ====
    columns_objects_tags_6 = [FIRST_STRING, TAG_COLOMN, PROJECT_COL]
    datasets_counts_6 = []
    # =========================================================================================== 7 ====
    columns_objects_tags_7 = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN, TOTAL_COL]
    datasets_counts_7 = []
    obj_tags_to_vals = defaultdict(list)
    # =========================================================================================== 8 ====
    columns_objects_tags_vals_8 = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN, PROJECT_COL]
    datasets_counts_8 = []
    # =========================================================================================== 9 ====
    columns_images_urls_to_img_tags_9 = [FIRST_STRING, IMAGE_COL, DATASET_NAME, TAG_COLOMN, TAG_VALUE_COLOMN]
    datasets_counts_9 = []
    # =========================================================================================== 10 ====
    columns_images_urls_to_obj_tags_10 = [FIRST_STRING, IMAGE_COL, DATASET_NAME, TAG_COLOMN, TAG_VALUE_COLOMN,
                                          NUM_OBJECTS]
    datasets_counts_10 = []
    # =========================================================================================== 11 ====
    columns_objects_tags_11 = [FIRST_STRING, TAG_COLOMN, OBJECT_CLASS, PROJECT_COL]
    datasets_counts_11 = []
    # =========================================================================================== 12 ====
    columns_objects_tags_12 = [FIRST_STRING, TAG_COLOMN, TAG_VALUE_COLOMN, OBJECT_CLASS, PROJECT_COL]
    datasets_counts_12 = []

    id_to_tagmeta = meta.tag_metas.get_id_mapping()

    for dataset in api.dataset.get_list(PROJECT_ID):
        columns_images_tags_1.extend([dataset.name])                                            # 1
        ds_images_tags_1 = defaultdict(int)                                                     # 1
        columns_images_tags_2.extend([dataset.name])                                            # 2
        ds_tags_to_imgs_urls_2 = defaultdict(list)                                              # 2
        columns_images_tags_3.extend([dataset.name])                                            # 3
        ds_images_tags_vals_3 = defaultdict(lambda: defaultdict(int))                           # 3
        columns_images_tags_vals_4.extend([dataset.name])                                       # 4
        ds_tags_to_imgs_urls_4 = defaultdict(lambda: defaultdict(list))                         # 4
        columns_objects_tags_5.extend([dataset.name])                                           # 5
        ds_objects_tags_5 = defaultdict(int)                                                    # 5
        columns_objects_tags_6.extend([dataset.name])                                           # 6
        ds_obj_tags_to_imgs_urls_6 = defaultdict(lambda: defaultdict(int))                      # 6
        columns_objects_tags_7.extend([dataset.name])                                           # 7
        ds_objects_tags_vals_7 = defaultdict(lambda: defaultdict(int))                          # 7
        columns_objects_tags_vals_8.extend([dataset.name])                                      # 8
        ds_tags_to_imgs_urls_8 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))     # 8
        imgs_urls_to_img_tags_9 = defaultdict(lambda: defaultdict(list))                        # 9
        imgs_urls_to_obj_tags_10 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))   # 10
        columns_objects_tags_11.extend([dataset.name])                                          # 11
        obj_tags_to_class_11 = defaultdict(lambda: defaultdict(int))                            # 11
        columns_objects_tags_12.extend([dataset.name])                                          # 12
        obj_tags_to_class_12 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))       # 12

        images = api.image.get_list(dataset.id)

        for batch in sly.batched(images, batch_size=10):
            image_ids = []
            for image_info in batch:
                image_ids.append(image_info.id)
                curr_image_tags = sly.TagCollection.from_api_response(image_info.tags, meta.tag_metas, id_to_tagmeta)

                process_images_tags_1(curr_image_tags, ds_images_tags_1, state)                    # 1
                process_images_tags_2(curr_image_tags, image_info, ds_tags_to_imgs_urls_2, state)  # 2
                process_images_tags_3(curr_image_tags, ds_images_tags_vals_3, tags_to_vals, state) # 3
                process_images_tags_4(curr_image_tags, image_info, ds_tags_to_imgs_urls_4, state)  # 4
                process_images_urls_to_img_tags_9(curr_image_tags, image_info, imgs_urls_to_img_tags_9)  # 9

            ann_infos = api.annotation.download_batch(dataset.id, image_ids)

            for idx, ann_info in enumerate(ann_infos):
                ann = sly.Annotation.from_json(ann_info.annotation, meta)
                curr_object_tags = get_objects_tags(ann)
                process_objects_tags_5(curr_object_tags, ds_objects_tags_5, state)                  # 5
                process_objects_tags_6(curr_object_tags, batch[idx], ds_obj_tags_to_imgs_urls_6, state)  # 6
                process_objects_tags_7(curr_object_tags, ds_objects_tags_vals_7, obj_tags_to_vals, state)  # 7
                process_objects_tags_8(curr_object_tags, batch[idx], ds_tags_to_imgs_urls_8, state)         # 8
                process_images_urls_to_obj_tags_10(curr_object_tags, batch[idx], imgs_urls_to_obj_tags_10)  # 10

                process_obj_tags_to_class_11(ann, obj_tags_to_class_11, state)                          # 11
                process_obj_tags_to_class_12(ann, obj_tags_to_class_12, state)                          # 12

        datasets_counts_1.append((dataset.name, ds_images_tags_1))                                 # 1
        datasets_counts_2.append((dataset.name, ds_tags_to_imgs_urls_2))                           # 2
        datasets_counts_3.append((dataset.name, ds_images_tags_vals_3))                            # 3
        datasets_counts_4.append((dataset.name, ds_tags_to_imgs_urls_4))                           # 4
        datasets_counts_5.append((dataset.name, ds_objects_tags_5))                                # 5
        datasets_counts_6.append((dataset.name, ds_obj_tags_to_imgs_urls_6))                       # 6
        datasets_counts_7.append((dataset.name, ds_objects_tags_vals_7))                           # 7
        datasets_counts_8.append((dataset.name, ds_tags_to_imgs_urls_8))                           # 8
        datasets_counts_9.append((dataset.name, imgs_urls_to_img_tags_9))                          # 9
        datasets_counts_10.append((dataset.name, imgs_urls_to_obj_tags_10))                        # 10
        datasets_counts_11.append((dataset.name, obj_tags_to_class_11))                            # 11
        datasets_counts_12.append((dataset.name, obj_tags_to_class_12))                            # 12

    df_1 = get_pd_tag_stat_1(datasets_counts_1, columns_images_tags_1, state)                      # 1
    print(df_1)                                                                                    # 1
    df_2 = get_pd_tag_stat_2(datasets_counts_2, columns_images_tags_2, state)                      # 2
    print(df_2)                                                                                    # 2
    df_3 = get_pd_tag_stat_3(datasets_counts_3, columns_images_tags_3, tags_to_vals)               # 3
    print(df_3)                                                                                    # 3
    df_4 = get_pd_tag_stat_4(datasets_counts_4, columns_images_tags_vals_4, tags_to_vals)          # 4
    print(df_4)                                                                                    # 4
    df_5 = get_pd_tag_stat_5(datasets_counts_5, columns_objects_tags_5, state)                     # 5
    print(df_5)                                                                                    # 5
    df_6 = get_pd_tag_stat_6(datasets_counts_6, columns_objects_tags_6, state)                     # 6
    print(df_6)                                                                                    # 6
    df_7 = get_pd_tag_stat_7(datasets_counts_7, columns_objects_tags_7, obj_tags_to_vals)          # 7
    print(df_7)                                                                                    # 7
    df_8 = get_pd_tag_stat_8(datasets_counts_8, columns_objects_tags_vals_8, obj_tags_to_vals)     # 8
    print(df_8)                                                                                    # 8
    df_9 = get_pd_tag_stat_9(datasets_counts_9, columns_images_urls_to_img_tags_9)                 # 9
    print(df_9)
    df_10 = get_pd_tag_stat_10(datasets_counts_10, columns_images_urls_to_obj_tags_10)             # 10
    print(df_10)
    df_11 = get_pd_tag_stat_11(meta, datasets_counts_11, columns_objects_tags_11, state)           # 11
    print(df_11)                                                                                   # 11
    df_12 = get_pd_tag_stat_12(meta, datasets_counts_12, columns_objects_tags_12, obj_tags_to_vals)  # 12
    print(df_12)                                                                                     # 12

    report_name = "{}_{}.lnk".format(PROJECT_ID, project_info.name)
    local_path = os.path.join(my_app.data_dir, report_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(my_app.app_url, file=text_file)
    remote_path = "/reports/images_tags_stat/{}".format(report_name)
    api.file.remove(TEAM_ID, remote_path)
    # remote_path = api.file.get_free_name(TEAM_ID, remote_path)
    report_name = sly.fs.get_file_name_with_ext(remote_path)
    file_info = api.file.upload(TEAM_ID, local_path, remote_path)
    report_url = api.file.get_url(file_info.id)

    update_tables = False

    if len(objects_tags) != 0 and len(state['choose_objs_tags']) == 0:
        table_11 = {"field": "state.ObjTagsNoExist", "payload": True}
        table_12 = {"field": "state.noObjectsAndVals", "payload": False}
    elif len(objects_tags) == 0:
        table_11 = {"field": "state.noObjects", "payload": False}
        table_12 = {"field": "state.noObjectsAndVals", "payload": False}
    else:
        table_11 = {"field": "data.obj_tags_to_classes_statTable", "payload": json.loads(df_11.to_json(orient="split"))}
        table_12 = {"field": "data.obj_tags_vals_to_classes_statTable", "payload": json.loads(df_12.to_json(orient="split"))}
        table_13 = {"field": "state.ObjTagsNoExist", "payload": False}
        table_14 = {"field": "state.noObjectsAndVals", "payload": True}
        update_tables = True

    fields = [
        {"field": "data.loading", "payload": False},
        {"field": "data.imgs_tags_statTable", "payload": json.loads(df_1.to_json(orient="split"))},
        {"field": "data.tags_to_imgs_urls_statTable", "payload": json.loads(df_2.to_json(orient="split"))},
        {"field": "data.imgs_tags_vals_statTable", "payload": json.loads(df_3.to_json(orient="split"))},
        {"field": "data.tags_vals_to_imgs_urls_statTable", "payload": json.loads(df_4.to_json(orient="split"))},
        {"field": "data.objs_tags_statTable", "payload": json.loads(df_5.to_json(orient="split"))},
        {"field": "data.obj_tags_to_imgs_urls_statTable", "payload": json.loads(df_6.to_json(orient="split"))},
        {"field": "data.objs_tags_vals_statTable", "payload": json.loads(df_7.to_json(orient="split"))},
        {"field": "data.obj_tags_vals_to_imgs_urls_statTable", "payload": json.loads(df_8.to_json(orient="split"))},
        {"field": "data.images_to_imgs_tag_val_statTable", "payload": json.loads(df_9.to_json(orient="split"))},
        {"field": "data.images_to_objs_tag_val_statTable", "payload": json.loads(df_10.to_json(orient="split"))},
        table_11,
        table_12,
        {"field": "data.savePath", "payload": remote_path},
        {"field": "data.reportName", "payload": report_name},
        {"field": "data.reportUrl", "payload": report_url},
    ]

    if update_tables:
        fields.extend([table_13, table_14])

    api.task.set_fields(task_id, fields)


@my_app.callback("choose_tags_values")
@sly.timeit
def choose_tags_values(api: sly.Api, task_id, context, state, app_logger):

    project_info = api.project.get_info_by_id(PROJECT_ID)
    meta_json = api.project.get_meta(project_info.id)
    meta = sly.ProjectMeta.from_json(meta_json)

    id_to_tagmeta = meta.tag_metas.get_id_mapping()

    for dataset in api.dataset.get_list(PROJECT_ID):
        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images, batch_size=10):
            image_ids = []
            for image_info in batch:
                image_ids.append(image_info.id)
                curr_image_tags = sly.TagCollection.from_api_response(image_info.tags, meta.tag_metas, id_to_tagmeta)
                for curr_tag in curr_image_tags:
                    if curr_tag.name in state['choose_tags']:
                        if curr_tag.value not in images_tags_to_values[curr_tag.name] and curr_tag.value is not None:
                            images_tags_to_values[curr_tag.name].append(curr_tag.value)

    select_data = []

    for tag_name in images_tags_to_values:
        options_data = []
        for tag_val in images_tags_to_values[tag_name]:
            options_data.append({"value": str(tag_val) + ' ' + tag_name + ' ' + type(tag_val).__name__, "label":tag_val})
        select_data.append({"label": tag_name, "options":options_data})

    fields = [
        {"field": "data.loading", "payload": False},
        {"field": "data.imgs_tags_statTable", "payload": []},
        {"field": "data.tags_to_imgs_urls_statTable", "payload": []},
        {"field": "data.imgs_tags_vals_statTable", "payload": []},
        {"field": "data.tags_vals_to_imgs_urls_statTable", "payload": []},
        {"field": "data.objs_tags_statTable", "payload": []},
        {"field": "data.obj_tags_to_imgs_urls_statTable", "payload": []},
        {"field": "data.objs_tags_vals_statTable", "payload": []},
        {"field": "data.obj_tags_vals_to_imgs_urls_statTable", "payload": []},
        {"field": "data.images_to_imgs_tag_val_statTable", "payload": []},
        {"field": "data.images_to_objs_tag_val_statTable", "payload": []},
        {"field": "data.obj_tags_to_classes_statTable", "payload": []},
        {"field": "data.obj_tags_vals_to_classes_statTable", "payload": []},
        {"field": "state.options3", "payload": select_data}
    ]
    api.task.set_fields(task_id, fields)


@my_app.callback("choose_objs_tags_values")
@sly.timeit
def choose_objs_tags_values(api: sly.Api, task_id, context, state, app_logger):

    project_info = api.project.get_info_by_id(PROJECT_ID)
    meta_json = api.project.get_meta(project_info.id)
    meta = sly.ProjectMeta.from_json(meta_json)

    for dataset in api.dataset.get_list(PROJECT_ID):
        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images, batch_size=10):
            image_ids = []
            for image_info in batch:
                image_ids.append(image_info.id)

            ann_infos = api.annotation.download_batch(dataset.id, image_ids)
            for idx, ann_info in enumerate(ann_infos):
                ann = sly.Annotation.from_json(ann_info.annotation, meta)
                curr_object_tags = get_objects_tags(ann)
                for curr_tag in curr_object_tags:
                    if curr_tag.name in state['choose_objs_tags']:
                        if curr_tag.value not in objects_tags_to_values[curr_tag.name] and curr_tag.value is not None:
                            objects_tags_to_values[curr_tag.name].append(curr_tag.value)

    select_data = []

    for tag_name in objects_tags_to_values:
        options_data = []
        for tag_val in objects_tags_to_values[tag_name]:
            options_data.append({"value": str(tag_val) + ' ' + tag_name + ' ' + type(tag_val).__name__, "label":tag_val})
        select_data.append({"label": tag_name, "options":options_data})

    fields = [
        {"field": "data.loading", "payload": False},
        {"field": "data.imgs_tags_statTable", "payload": []},
        {"field": "data.tags_to_imgs_urls_statTable", "payload": []},
        {"field": "data.imgs_tags_vals_statTable", "payload": []},
        {"field": "data.tags_vals_to_imgs_urls_statTable", "payload": []},
        {"field": "data.objs_tags_statTable", "payload": []},
        {"field": "data.obj_tags_to_imgs_urls_statTable", "payload": []},
        {"field": "data.objs_tags_vals_statTable", "payload": []},
        {"field": "data.obj_tags_vals_to_imgs_urls_statTable", "payload": []},
        {"field": "data.images_to_imgs_tag_val_statTable", "payload": []},
        {"field": "data.images_to_objs_tag_val_statTable", "payload": []},
        {"field": "data.obj_tags_to_classes_statTable", "payload": []},
        {"field": "data.obj_tags_vals_to_classes_statTable", "payload": []},
        {"field": "state.options3_objs", "payload": select_data}
    ]
    api.task.set_fields(task_id, fields)


@my_app.callback("images_tags_stats")
@sly.timeit
def images_tags_stats(api: sly.Api, task_id, context, state, app_logger):

    project_info = api.project.get_info_by_id(PROJECT_ID)
    meta_json = api.project.get_meta(project_info.id)
    meta = sly.ProjectMeta.from_json(meta_json)

    images_tags = []
    for tag in meta.tag_metas:
        if tag.applicable_to != TagApplicableTo.OBJECTS_ONLY:
            images_tags.append({"value": tag.name, "label": tag.name})
        if tag.applicable_to != TagApplicableTo.IMAGES_ONLY:
            objects_tags.append({"value": tag.name, "label": tag.name})

    fields = [
        {"field": "data.loading", "payload": False},
        {"field": "data.imgs_tags_statTable", "payload": []},
        {"field": "data.tags_to_imgs_urls_statTable", "payload": []},
        {"field": "data.imgs_tags_vals_statTable", "payload": []},
        {"field": "data.tags_vals_to_imgs_urls_statTable", "payload":[]},
        {"field": "data.objs_tags_statTable", "payload": []},
        {"field": "data.obj_tags_to_imgs_urls_statTable", "payload": []},
        {"field": "data.objs_tags_vals_statTable", "payload": []},
        {"field": "data.obj_tags_vals_to_imgs_urls_statTable", "payload": []},
        {"field": "data.images_to_imgs_tag_val_statTable", "payload": []},
        {"field": "data.images_to_objs_tag_val_statTable", "payload": []},
        {"field": "data.obj_tags_to_classes_statTable", "payload": []},
        {"field": "data.obj_tags_vals_to_classes_statTable", "payload": []},
        {"field": "state.options", "payload": images_tags},
        {"field": "state.options_objs", "payload": objects_tags},

        {"field": "data.projectName", "payload": project_info.name},
        {"field": "data.projectId", "payload": project_info.id},
        {"field": "data.projectPreviewUrl", "payload": api.image.preview_url(project_info.reference_image_url, 100, 100)},
    ]
    api.task.set_fields(task_id, fields)


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": TEAM_ID,
        "WORKSPACE_ID": WORKSPACE_ID,
        "PROJECT_ID": PROJECT_ID
    })

    data = {
        "userImageTable": {"columns": [], "data": []}
    }
    state = {"noObjects": True, "noObjectsAndVals":  True}

    my_app.run(data=data, state=state, initial_events=[{"command": "images_tags_stats"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
