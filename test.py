import os


def gen_train_txt(txt_path, img_dir):

    f = open(txt_path, 'w')
    img_list = os.listdir(img_dir)  # 获取类别文件夹下所有png图片的路径
    print(len(img_list))
    for i in range(38000):
        label_path = os.path.join("../wh_dataset/", img_list[i])
        # img_path = os.path.join("../LOLdataset/our485/img/", img_list[i])
        line = label_path + '\n'
        print(line)
        f.write(line)
    f.close()


if __name__ == '__main__':
    gen_train_txt('./data.txt', '../wh_dataset')
