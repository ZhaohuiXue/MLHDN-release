# generate train and test samples script
"""
@author: mengxue.zhang
"""


import os
import numpy as np
import math
import scipy.io as sio


# constant values
train_base_dir = './MSTAR-10/train/'
test_base_dir = './MSTAR-10/test/'
eoc2_base_dir = './MSTAR-10/eoc/'
mat_base_dir = './input'
random_obj = np.random.RandomState(seed=0) # 2020
verbose = True
class_name = {'2S1':0, 'BMP2':1, 'BRDM_2':2, 'BTR60':3, 'BTR70':4, 'D7':5, 'T62':6, 'T72':7, 'ZIL131':8, 'ZSU_23_4':9}


def list_dir_or_file(type='dir', base_path='./MSTAR-10/train/', key='jpeg'):
    list_path = os.listdir(base_path)
    list_result = []
    for item in list_path:
        child = os.path.join(base_path, item)
        if type == 'dir':
            if os.path.isdir(child):
                list_result.append(child)
        elif type == 'file':
            if child.find(key)!=-1:
                if os.path.isfile(child):
                    list_result.append(child)
        else:
            print('Invalid type parameter calling list_dir_or_file!')

    if key == 'txt':
        results = list_result
        list_result = []
        for item in results:
            with open(item, 'r') as f:
                info = f.readline()
            list_result.append(info)

    return list_result


def gen_dict_and_store(images_path, aspect, class_index, name='undefined'):
    if not os.path.exists(mat_base_dir):
        os.mkdir(mat_base_dir)
    file_name = os.path.join(mat_base_dir, name + '.mat')
    sio.savemat(file_name, {'images': images_path, 'aspect' : aspect, 'labels': class_index}, format='5')


def search_other_idx(l_angles, l_idx, ai, view=3, num=99999, theta=360, fnum=99999):
    def generate_multi_views(sorted_idx, l_idx, view=3):
        selected_set = []
        x = l_idx[ai]

        rii = []
        for i in range(view):
            ri = np.arange(sorted_idx.size)
            random_obj.shuffle(ri)
            rii.append(ri[:num])
        if view == 0:
            selected_set.append([x])
        elif view == 1:
            for a in rii[0]:
                selected_set.append([x, sorted_idx[a]])
        elif view == 2:
            for a in rii[0]:
                for b in rii[1]:
                    if a == b:
                        continue
                    selected_set.append([x, sorted_idx[a], sorted_idx[b]])
        # elif view == 3:
        #     for a in rii[0]:
        #         for b in rii[1]:
        #             if a == b:
        #                 continue;
        #             for c in rii[2]:
        #                 if c == b or c == a:
        #                     continue;
        #                 selected_set.append([x, sorted_idx[a], sorted_idx[b], sorted_idx[c]])
        elif view == 3:
            for a in rii[0]:
                for b in rii[1]:
                    if a == b:
                        continue
                    for c in rii[2]:
                        if c == b or c == a:
                            continue
                        selected_set.append([x, sorted_idx[a], sorted_idx[b], sorted_idx[c]])

        elif view == 4:
            for a in rii[0]:
                for b in rii[1]:
                    if a == b:
                        continue
                    for c in rii[2]:
                        if c == b or c == a:
                            continue
                        for d in rii[3]:
                            if d == c or d == b or d == a:
                                continue
                            selected_set.append([x, sorted_idx[a], sorted_idx[b], sorted_idx[c], sorted_idx[d]])

        elif view == 5:
            for a in rii[0]:
                for b in rii[1]:
                    if a == b:
                        continue
                    for c in rii[2]:
                        if c == b or c == a:
                            continue
                        for d in rii[3]:
                            if d == c or d == b or d == a:
                                continue
                            for e in rii[4]:
                                if e == d or e == c or e == b or e == a:
                                    continue
                                selected_set.append([x, sorted_idx[a], sorted_idx[b], sorted_idx[c], sorted_idx[d], sorted_idx[e]])

        elif view == 6:
            for a in rii[0]:
                for b in rii[1]:
                    if a == b:
                        continue
                    for c in rii[2]:
                        if c == b or c == a:
                            continue
                        for d in rii[3]:
                            if d == c or d == b or d == a:
                                continue
                            for e in rii[4]:
                                if e == d or e == c or e == b or e == a:
                                    continue
                                for f in rii[5]:
                                    if f == e or f == d or f == c or f == b or f == a:
                                        continue
                                    selected_set.append(
                                        [x, sorted_idx[a], sorted_idx[b], sorted_idx[c], sorted_idx[d], sorted_idx[e], sorted_idx[f]])

        elif view == 7:
            for a in rii[0]:
                for b in rii[1]:
                    if a == b:
                        continue
                    for c in rii[2]:
                        if c == b or c == a:
                            continue
                        for d in rii[3]:
                            if d == c or d == b or d == a:
                                continue
                            for e in rii[4]:
                                if e == d or e == c or e == b or e == a:
                                    continue
                                for f in rii[5]:
                                    if f == e or f == d or f == c or f == b or f == a:
                                        continue
                                    for g in rii[6]:
                                        if g == f or g == e or g == d or g == c or g == b or g == a:
                                            continue
                                        selected_set.append(
                                        [x, sorted_idx[a], sorted_idx[b], sorted_idx[c], sorted_idx[d], sorted_idx[e],
                                         sorted_idx[f], sorted_idx[g]])

        elif view == 8:
            for a in rii[0]:
                for b in rii[1]:
                    if a == b:
                        continue
                    for c in rii[2]:
                        if c == b or c == a:
                            continue
                        for d in rii[3]:
                            if d == c or d == b or d == a:
                                continue
                            for e in rii[4]:
                                if e == d or e == c or e == b or e == a:
                                    continue
                                for f in rii[5]:
                                    if f == e or f == d or f == c or f == b or f == a:
                                        continue
                                    for g in rii[6]:
                                        if g == f or g == e or g == d or g == c or g == b or g == a:
                                            continue
                                        for h in rii[7]:
                                            if h == g or h == f or h == e or h == d or h == c or h == b or h == a:
                                                continue
                                            selected_set.append(
                                                [x, sorted_idx[a], sorted_idx[b], sorted_idx[c], sorted_idx[d],
                                                 sorted_idx[e],
                                                 sorted_idx[f], sorted_idx[g], sorted_idx[h]])

            selected_set = np.array(selected_set)
            r = np.arange(selected_set.shape[0]); random_obj.shuffle(r)
            selected_set = selected_set[r[:fnum]].tolist()
        return selected_set

    selected_set = []
    angles = []
    a_idx = []
    angle = l_angles[ai]
    max_angle = angle + theta

    idx = np.arange(l_angles.size)
    for i in idx:
        ag = l_angles[i]
        if max_angle > 360:
            if ag > angle or ag < max_angle - 360:
                a_idx.append(l_idx[i])
                angles.append(ag)
        else:
            if ag > angle and ag < max_angle:
                a_idx.append(l_idx[i])
                angles.append(ag)

    if len(angles) >= view:
        sort_i = np.argsort(angles)
        sorted_idx = np.array(a_idx)[sort_i]
        selected_set = generate_multi_views(sorted_idx, l_idx, view)
        return selected_set
    else:
        return []


def generate_set_from_selected_idx(l_angles, selected_idx, view=4, maximum=99999, balanced=False, theta=360):
    selected_angles = l_angles[selected_idx]
    selected_set = []
    selected_list = []
    selected_set_s = []

    for i in range(len(selected_idx)):
        if balanced:
            # when use balanced, we do not generate all possible pairs which can speed this code and make sampling process balanced
            fina_num = math.ceil(maximum / selected_idx.shape[0]) * 2
            mean_num = math.ceil(math.pow(maximum, 1 / view)) + 1
            combine_idx = search_other_idx(selected_angles, selected_idx, i, view=view-1, num=mean_num, theta=theta, fnum=fina_num)

            random_index = np.arange(0, 10)
            random_obj.shuffle(random_index)
            combine_idx = (np.array(combine_idx)[random_index]).tolist()
        else:
            combine_idx = search_other_idx(selected_angles, selected_idx, i, view=view-1, theta=theta, fnum=fina_num)

        if combine_idx:
            selected_set_s.append(len(combine_idx))
            selected_list.append(np.array(combine_idx))
            selected_set.extend(np.array(combine_idx))

    # remove same idx
    a_idx = []
    for item in selected_set:
        flag = False
        for a_item in a_idx:
            if set(a_item) == set(item):
                flag = True
                break
        if flag:
            continue
        else:
            a_idx.append(item)
    selected_set = a_idx

    selected_set = np.array(selected_set)
    # sum_2 = np.sum(np.array(selected_set_s))
    sum_2 = selected_set.shape[0]
    if maximum > sum_2:
        maximum = sum_2
    random_index = np.arange(sum_2)
    random_obj.shuffle(random_index)
    selected_set_idx = selected_set[random_index[:maximum]]
    return selected_set_idx


def process_train(times=1, class_names=class_name, base_path=train_base_dir,
                  prefix="", angle_noise=0, train_ratio=0.05, view=4, test_theta=360, train_type='train'):
    list_dir = list_dir_or_file(type='dir', base_path=base_path)

    for time in range(times):
        train_image_path = []
        valid_image_path = []
        train_angle = []
        valid_angle = []
        train_indexes = []
        valid_indexes = []
        for idx in range(len(list_dir)):
            d_item = list_dir[idx]
            name = d_item[d_item.rfind('/')+1:]
            if name not in class_names:
                continue

            label_index = class_names[name]
            l_images = list_dir_or_file(type='file', base_path=d_item, key='jpeg')

            l_angles = list_dir_or_file(type='file', base_path=d_item, key='txt')
            l_angles = np.array(l_angles, dtype=np.float32)

            for i in range(len(l_angles)):
                noise = random_obj.normal(loc=0, scale=angle_noise)
                l_angles[i] = l_angles[i] + noise

            sum_num = len(l_images)
            if train_ratio < 1:
                train_num = math.ceil(sum_num * train_ratio)
            else:
                train_num = int(train_ratio)

            random_index = np.arange(sum_num)
            random_obj.shuffle(random_index)
            train_selected_idx = random_index[:train_num]
            valid_selected_idx = random_index[train_num:]

            l_images = np.array(l_images)

            # maximum = int(math.ceil(43533 / 2747 * sum_num))
            maximum = 2000
            train_set_idx = generate_set_from_selected_idx(l_angles, train_selected_idx, view=view, maximum=maximum, balanced=True)
            # assert(train_set_idx.unique())
            assert(len(np.unique(train_set_idx.flatten()))==train_num)
            train_set_images = l_images[train_set_idx]
            train_set_angles = l_angles[train_set_idx]
            train_set_labels = np.ones(shape=[train_set_idx.shape[0], 1]) * label_index

            # 生成mat文件（包含样本对和各个方位角）
            train_image_path.append(train_set_images)
            train_angle.append(train_set_angles)
            train_indexes.append(train_set_labels)
            print(str(label_index) + ' ' + d_item + ': train ' + str(train_num))

            # maximum = 2000 # 2000
            # valid_set_idx = generate_set_from_selected_idx(l_angles, valid_selected_idx, view=view, maximum=maximum, balanced=True, theta=test_theta)
            # valid_set_images = l_images[valid_set_idx]
            # valid_set_angles = l_angles[valid_set_idx]
            # valid_set_labels = np.ones(shape=[valid_set_idx.shape[0], 1]) * label_index
            #
            # valid_image_path.append(valid_set_images)
            # valid_angle.append(valid_set_angles)
            # valid_indexes.append(valid_set_labels)
            # print(str(label_index) + ' ' + d_item + ': valid ' + str(sum_num-train_num))
            print('--------------------------')

        if train_type == 'train':
            name = 'train' + prefix + '_r' + str(time)
        elif train_type == 'view':
            name = 'train' + prefix + '_r' + str(time) + '_v' + str(view)
        else:
            print('invalid train_type parameters, using train_type = train')
            name = 'train' + prefix + '_r' + str(time)

        gen_dict_and_store(name=name, images_path=train_image_path, aspect=train_angle, class_index=train_indexes)
        # gen_dict_and_store(name='valid' + prefix, images_path=valid_image_path, aspect=valid_angle, class_index=valid_indexes)

    return None
    # return train_image_path, train_angle, train_indexes, valid_image_path, valid_angle, valid_indexes


def process_test(times=1, class_names=class_name, base_path=test_base_dir,
                 prefix="", angle_noise=0, view=4, test_theta=360, test_type='test'):
    list_dir = list_dir_or_file(type='dir', base_path=base_path)

    for time in range(times):
        test_image_path = []
        test_angle = []
        test_indexes = []
        for idx in range(len(list_dir)):
            d_item = list_dir[idx]
            name = d_item[d_item.rfind('/') + 1:]
            if name not in class_names:
                continue

            label_index = class_names[name]
            l_images = list_dir_or_file(type='file', base_path=d_item, key='jpeg')

            l_angles = list_dir_or_file(type='file', base_path=d_item, key='txt')
            l_angles = np.array(l_angles, dtype=np.float32)

            for i in range(len(l_angles)):
                noise = random_obj.normal(loc=0, scale=angle_noise)
                l_angles[i] = l_angles[i] + noise

            sum_num = len(l_images)
            random_index = np.arange(sum_num)
            random_obj.shuffle(random_index)

            test_selected_idx = random_index[:]
            l_images = np.array(l_images)

            maximum = 2000 # 2000
            test_set_idx = generate_set_from_selected_idx(l_angles, test_selected_idx, view=view, maximum=maximum,
                                                           balanced=True, theta=test_theta)
            # print(len(np.unique(test_set_idx.flatten())))
            assert (len(np.unique(test_set_idx.flatten())) == sum_num)
            test_set_images = l_images[test_set_idx]
            test_set_angles = l_angles[test_set_idx]
            test_set_labels = np.ones(shape=[test_set_idx.shape[0], 1]) * label_index

            test_image_path.append(test_set_images)
            test_angle.append(test_set_angles)
            test_indexes.append(test_set_labels)

            print(str(label_index) + ' ' + d_item + ': test ' + str(sum_num))
            print('--------------------------')

        if test_type == 'test':
            name = 'test' + prefix + '_r' + str(time)
        elif test_type == 'noise':
            name = 'test' + prefix + '_r' + str(time) + '_n' + str(angle_noise)
        elif test_type == 'theta':
            name = 'test' + prefix + '_r' + str(time) + '_t' + str(test_theta)
        elif test_type == 'view':
            name = 'test' + prefix + '_r' + str(time) + '_v' + str(view)
        else:
            print('invalid test_type parameters, using test_type = test')
            name = 'test' + prefix + '_r' + str(time)
        gen_dict_and_store(name=name, images_path=test_image_path, aspect=test_angle, class_index=test_indexes)

    return None
    # return test_image_path, test_angle, test_indexes


process_train(times=10, prefix='_10', angle_noise=0, view=4, train_ratio=10)
process_test(times=10, prefix='', angle_noise=0, view=4, test_theta=45, test_type='test')

# for angle_noise in range(1, 16):
#     process_test(times=10, prefix='', angle_noise=angle_noise, view=4, test_theta=45, test_type='noise')

# for test_theta in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
#     process_test(times=10, prefix='', angle_noise=0, view=4, test_theta=test_theta, test_type='theta')

# for view in range(2, 10):
#     process_train(times=10, prefix='_10', angle_noise=0, view=view, train_ratio=10)
#     process_test(times=10, prefix='', angle_noise=0, view=view, test_theta=45, test_type='view')

