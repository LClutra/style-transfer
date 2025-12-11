from __future__ import print_function

import imageio
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import scipy.io
import os

# TensorFlow 2.x 导入
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=2000, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')
parser.add_argument('--init_noise', type=float, default=0.0, required=False,
                    help='Initial noise level for generated image.')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter
init_noise = args.init_noise

# 不同损失分量的权重
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# 生成图像的尺寸
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# 创建保存结果的目录
output_dir = 'neural_style_transfer_keras'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# 工具函数：打开、调整大小并将图片格式化为适当的张量
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # 注意：原始的VGG19模型期望BGR格式，减去ImageNet均值
    # 但我们要确保这与MATLAB权重匹配
    img = vgg19.preprocess_input(img)
    return img

# 工具函数：将张量转换为有效的图像
def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    # 反转VGG19预处理
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[..., ::-1]  # BGR -> RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 加载 MATLAB 权重文件（如果需要的话）
def load_vgg_weights(mat_path):
    """加载 MATLAB 格式的 VGG19 权重"""
    try:
        data = scipy.io.loadmat(mat_path)
        layers = data['layers'][0]
        
        weights = {}
        for i, layer in enumerate(layers):
            layer_name = layer[0][0][0][0]
            layer_type = layer[0][0][1][0]
            
            if layer_type == 'conv':
                # 获取权重和偏置
                w = layer[0][0][2][0][0]
                b = layer[0][0][2][0][1]
                
                # 调整权重维度顺序：MATLAB [H, W, C_in, C_out] -> Keras [H, W, C_in, C_out]
                # 注意：MATLAB存储的是列优先，需要转置
                w = np.transpose(w, (1, 0, 2, 3))
                
                weights[layer_name] = [w, b.flatten()]
        
        return weights
    except FileNotFoundError:
        print(f"Warning: MATLAB weights file '{mat_path}' not found. Using Keras pretrained weights.")
        return None
    except Exception as e:
        print(f"Error loading MATLAB weights: {e}. Using Keras pretrained weights.")
        return None

# 创建 VGG19 模型
def build_vgg19_model():
    """构建 VGG19 模型，优先使用MATLAB权重，否则使用Keras权重"""
    # 尝试加载MATLAB权重
    matlab_weights = load_vgg_weights('imagenet-vgg-verydeep-19.mat')
    
    # 构建模型结构
    input_tensor = Input(shape=(img_nrows, img_ncols, 3))
    
    # 使用Keras的VGG19模型，但不包括顶层（分类层）
    model = vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
    
    # 如果成功加载了MATLAB权重，则替换Keras权重
    if matlab_weights:
        print("Loading MATLAB weights...")
        for layer in model.layers:
            if layer.name in matlab_weights:
                try:
                    layer.set_weights(matlab_weights[layer.name])
                    print(f'Loaded weights for {layer.name}')
                except Exception as e:
                    print(f'Warning: Could not load weights for {layer.name}: {e}')
        print("MATLAB weights loaded successfully.")
    else:
        print("Using Keras pretrained weights.")
    
    # 冻结所有层，因为我们在风格迁移中不需要训练
    for layer in model.layers:
        layer.trainable = False
    
    return model

# 获取预处理图像
base_image = K.constant(preprocess_image(base_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# 初始化生成图像：使用内容图像+噪声
base_image_np = preprocess_image(base_image_path)
if init_noise > 0:
    combination_image = K.variable(base_image_np + np.random.normal(0, init_noise, base_image_np.shape))
    print(f"Initialized with content image + noise (σ={init_noise})")
else:
    combination_image = K.variable(base_image_np)
    print("Initialized with content image")

# 构建模型
model = build_vgg19_model()
print('Model loaded.')

# 获取模型输出
outputs_dict = {layer.name: layer.output for layer in model.layers}

# 创建损失模型
layer_names = ['block5_conv2',  # 内容层
               'block1_conv1', 'block2_conv1',
               'block3_conv1', 'block4_conv1',
               'block5_conv1']  # 风格层

# 确保这些层存在于模型中
existing_layers = [layer for layer in layer_names if layer in outputs_dict]
if len(existing_layers) != len(layer_names):
    print(f"Warning: Some layers not found. Using existing layers: {existing_layers}")

loss_model = Model(inputs=model.input, outputs=[outputs_dict[layer] for layer in existing_layers])

# 计算神经风格损失
def gram_matrix(x):
    # 确保x是3D张量 (height, width, channels)
    if K.ndim(x) == 4:
        # 如果是批处理，取第一个元素
        x = x[0, :, :, :]
    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def total_variation_loss(x):
    if K.ndim(x) == 4:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    else:
        a = K.square(x[:img_nrows - 1, :img_ncols - 1, :] - x[1:, :img_ncols - 1, :])
        b = K.square(x[:img_nrows - 1, :img_ncols - 1, :] - x[:img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# 计算总损失
def total_loss(generated_image):
    # 确保generated_image是正确形状的张量
    if K.ndim(generated_image) == 1:
        generated_batch = K.reshape(generated_image, (1, img_nrows, img_ncols, 3))
    else:
        generated_batch = generated_image
    
    # 确保base_image和style_reference_image是正确形状
    if K.ndim(base_image) == 4:
        base_batch = base_image
    else:
        base_batch = K.reshape(base_image, (1, img_nrows, img_ncols, 3))
    
    if K.ndim(style_reference_image) == 4:
        style_batch = style_reference_image
    else:
        style_batch = K.reshape(style_reference_image, (1, img_nrows, img_ncols, 3))
    
    # 拼接批次
    input_batch = K.concatenate([base_batch, style_batch, generated_batch], axis=0)
    
    # 获取模型输出
    outputs = loss_model(input_batch)
    
    # 计算内容损失（使用block5_conv2）
    content_layer_name = 'block5_conv2'
    if content_layer_name in existing_layers:
        content_idx = existing_layers.index(content_layer_name)
        content_features = outputs[content_idx][0]  # 第一个是内容图像
        combination_features = outputs[content_idx][2]  # 第三个是生成图像
        c_loss = content_weight * content_loss(content_features, combination_features)
    else:
        c_loss = K.constant(0.0)
        print(f"Warning: Content layer '{content_layer_name}' not found. Content loss set to 0.")
    
    # 计算风格损失
    style_losses = []
    style_layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    for layer_name in style_layer_names:
        if layer_name in existing_layers:
            layer_idx = existing_layers.index(layer_name)
            layer_features = outputs[layer_idx]
            style_reference_features = layer_features[1]  # 第二个是风格图像
            combination_features = layer_features[2]  # 第三个是生成图像
            sl = style_loss(style_reference_features, combination_features)
            style_losses.append((style_weight / len(style_layer_names)) * sl)
    
    s_loss = K.sum(style_losses) if style_losses else K.constant(0.0)
    
    # 计算总变差损失
    tv_loss = total_variation_weight * total_variation_loss(generated_batch)
    
    # 总损失
    total = c_loss + s_loss + tv_loss
    return total

# 创建函数来计算损失和梯度
def compute_loss_and_grads(generated_image):
    # 确保generated_image是变量
    if not isinstance(generated_image, tf.Variable):
        generated_image_var = tf.Variable(generated_image)
    else:
        generated_image_var = generated_image
    
    with tf.GradientTape() as tape:
        tape.watch(generated_image_var)
        loss_value = total_loss(generated_image_var)
    
    # 计算梯度
    grads = tape.gradient(loss_value, generated_image_var)
    
    # 如果梯度为None，设置为零
    if grads is None:
        print("Warning: Gradient is None, setting to zero")
        grads = tf.zeros_like(generated_image_var)
    
    # 确保梯度不是NaN或Inf
    grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
    grads = tf.where(tf.math.is_inf(grads), tf.zeros_like(grads), grads)
    
    return loss_value, grads

# Evaluator 类 - 修改为保存回调函数
class Evaluator:
    def __init__(self, callback=None):
        self.loss_value = None
        self.grads_values = None
        self.callback = callback
        self.iteration = 0
    
    def loss(self, x):
        x_reshaped = x.reshape((1, img_nrows, img_ncols, 3))
        x_tensor = tf.convert_to_tensor(x_reshaped, dtype=tf.float32)
        
        loss_value, grads = compute_loss_and_grads(x_tensor)
        
        self.loss_value = float(loss_value.numpy())
        grad_numpy = grads.numpy()
        
        # 检查梯度是否全为零
        grad_norm = np.linalg.norm(grad_numpy)
        if grad_norm < 1e-10:
            print(f"Warning: Gradient norm is very small: {grad_norm}")
        
        self.grads_values = grad_numpy.flatten().astype('float64')
        
        # 调用回调函数
        self.iteration += 1
        if self.callback:
            self.callback(self.iteration, x, self.loss_value, grad_norm)
        
        return self.loss_value
    
    def grads(self, x):
        if self.grads_values is None:
            # 如果grads_values为None，重新计算
            _ = self.loss(x)
        
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values

# 初始化生成图像
x = preprocess_image(base_image_path)
if init_noise > 0:
    x = x + np.random.normal(0, init_noise, x.shape)

# 用于保存中间结果的回调函数
def save_callback(iteration, x_current, loss_value, grad_norm):
    # 每隔50次迭代保存一次图像
    if iteration % 50 == 0:
        img = deprocess_image(x_current.copy())
        fname = os.path.join(output_dir, f'{result_prefix}_iteration_{iteration:04d}.png')
        imageio.imwrite(fname, img)
        print(f'Iteration {iteration}: Loss = {loss_value:.2f}, Grad norm = {grad_norm:.6f}, Image saved')
    elif iteration % 10 == 0:  # 每10次打印一次损失
        print(f'Iteration {iteration}: Loss = {loss_value:.2f}, Grad norm = {grad_norm:.6f}')

# 创建带回调的评估器
evaluator = Evaluator(callback=save_callback)

print(f'Starting optimization for {iterations} iterations...')
print(f'Image size: {img_nrows}x{img_ncols}')
print(f'Weights: content={content_weight}, style={style_weight}, tv={total_variation_weight}')

# 使用 fmin_l_bfgs_b 一次性运行所有迭代，调整参数避免过早收敛
start_time = time.time()

# 关键修改：调整优化器参数
x_optimized, min_val, info = fmin_l_bfgs_b(
    evaluator.loss, 
    x.flatten(),
    fprime=evaluator.grads, 
    maxfun=iterations,  # 设置总迭代次数
    factr=1e7,  # 收敛精度（值越大越宽松，默认1e7）
    pgtol=1e-10,  # 梯度容差（默认是1e-5）
    maxls=50,  # 最大线搜索次数
    iprint=50,  # 每50次迭代打印一次
    maxiter=iterations  # 最大迭代次数
)

total_time = time.time() - start_time
print(f'Optimization completed in {total_time:.2f} seconds')
print(f'Optimization info: {info}')
print(f'Final loss value: {min_val}')

# 保存最终图像
img = deprocess_image(x_optimized.copy())
fname = os.path.join(output_dir, f'{result_prefix}_final.png')
imageio.imwrite(fname, img)
print(f'Final image saved as {fname}')
print('Style transfer completed!')

# 额外保存一个包含所有参数信息的版本
info_fname = os.path.join(output_dir, f'{result_prefix}_info.txt')
with open(info_fname, 'w') as f:
    f.write(f'Base image: {base_image_path}\n')
    f.write(f'Style image: {style_reference_image_path}\n')
    f.write(f'Iterations: {iterations}\n')
    f.write(f'Content weight: {content_weight}\n')
    f.write(f'Style weight: {style_weight}\n')
    f.write(f'TV weight: {total_variation_weight}\n')
    f.write(f'Image size: {img_nrows}x{img_ncols}\n')
    f.write(f'Initial noise: {init_noise}\n')
    f.write(f'Final loss: {min_val}\n')
    f.write(f'Optimization time: {total_time:.2f} seconds\n')
    f.write(f'Optimization info: {info}\n')
print(f'Parameters saved to {info_fname}')