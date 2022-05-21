import numpy as np


# 生成基础的先验框
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    # 用0填充9个先验框
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    # 两层for循环，每个循环长度为3，可以获得9个基础的先验框
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


# 对基础先验框进行拓展对应到所有特征点上
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):

    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shift.shape=(1444, 4)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]  # 38*38*9
    K = shift.shape[0]  # 38*38
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))  # anchor.shape=(1444, 9, 4)
     
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  # anchor.shape=(12996, 4)
    return anchor


if __name__ == "__main__":
    # 打印9个先验框
    nine_anchors = generate_anchor_base()
    print(nine_anchors)

    # 将先验框映射到特征点
    import matplotlib.pyplot as plt
    height, width, feat_stride = 38, 38, 16  # feat_stride=16:表示原图上每16个像素点有一个网格
    anchors_all = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)  # anchor_all.shape=(12996, 4)
    
    fig = plt.figure()
    # 规划figure划分子图
    ax = fig.add_subplot(111)
    # 同时plt.ylim() 显示的是y轴的作图范围
    plt.ylim(-300, 900)
    # plt.xlim() 显示的是x轴的作图范围
    plt.xlim(-300, 900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)  # shift_x.shape=(38,)
    shift_y = np.arange(0, height * feat_stride, feat_stride)  # shift_y.shape=(38,)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 利用shift_x和shift_y生成坐标点
    # 用于生成一个scatter散点图
    plt.scatter(shift_x, shift_y)

    box_widths = anchors_all[:, 2]-anchors_all[:, 0]  # box_widths.shape=(12996,)
    box_heights = anchors_all[:, 3]-anchors_all[:, 1]  # box_heights.shape=(12996,)
    
    for i in [208, 209, 210, 211, 212, 213, 214, 215, 216]:
        rect = plt.Rectangle([anchors_all[i, 0], anchors_all[i, 1]],
                             box_widths[i], box_heights[i], color="r", fill=False)
        ax.add_patch(rect)
    plt.show()
