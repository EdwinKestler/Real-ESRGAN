import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Define the local paths to the pre-trained models
MODEL_PATHS = {
    'RealESRGAN_x4plus': '/content/gdrive/MyDrive/Real-ESRGAN-Inference-models/pretrained_models/RealESRGAN_x4plus.pth',
    'RealESRNet_x4plus': '/content/gdrive/MyDrive/Real-ESRGAN-Inference-models/pretrained_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
    'RealESRGAN_x4plus_anime_6B': '/content/gdrive/MyDrive/Real-ESRGAN-Inference-models/pretrained_models/RealESRGAN_x4plus_anime_6B.pth',
    'RealESRGAN_x2plus': '/content/gdrive/MyDrive/Real-ESRGAN-Inference-models/pretrained_models/RealESRGAN_x2plus.pth',
    'RealESRGAN_x4plus_netD': '/content/gdrive/MyDrive/Real-ESRGAN-Inference-models/pretrained_models/RealESRGAN_x4plus_netD.pth',
    'realesr-general-x4v3': '/content/gdrive/MyDrive/Real-ESRGAN-Inference-models/pretrained_models/realesr-general-x4v3.pth',
    'realesr-general-wdn-x4v3': '/content/gdrive/MyDrive/Real-ESRGAN-Inference-models/pretrained_models/realesr-general-wdn-x4v3.pth'
}

def get_model_config(model_name):
    model_configs = {
        'RealESRGAN_x4plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'netscale': 4,
            'model_path': MODEL_PATHS['RealESRGAN_x4plus']
        },
        'RealESRNet_x4plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'netscale': 4,
            'model_path': MODEL_PATHS['RealESRNet_x4plus']
        },
        'RealESRGAN_x4plus_anime_6B': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
            'netscale': 4,
            'model_path': MODEL_PATHS['RealESRGAN_x4plus_anime_6B']
        },
        'RealESRGAN_x2plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            'netscale': 2,
            'model_path': MODEL_PATHS['RealESRGAN_x2plus']
        },
        'RealESRGAN_x4plus_netD': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'netscale': 4,
            'model_path': MODEL_PATHS['RealESRGAN_x4plus_netD']
        },
        'realesr-animevideov3': {
            'model': SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
            'netscale': 4,
            'model_path': MODEL_PATHS['realesr-general-x4v3']
        },
        'realesr-general-x4v3': {
            'model': SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
            'netscale': 4,
            'model_path': MODEL_PATHS['realesr-general-x4v3']
        }
    }
    return model_configs.get(model_name.split('.')[0], None)
    
def main():
    parser = argparse.ArgumentParser(description="Inference demo for Real-ESRGAN")
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5, help='Denoise strength for the general-x4v3 model')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='Final upsampling scale of the image')
    parser.add_argument('--model_path', type=str, default=None, help='Optional model path')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference')
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan', help='Alpha upsampler (realesrgan | bicubic)')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension (auto | jpg | png)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help='GPU device ID to use (default=0)')

    args = parser.parse_args()

    model_config = get_model_config(args.model_name)
    if model_config is None:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    model = model_config['model']
    netscale = model_config['netscale']
    model_path = args.model_path or model_config['model_path']

    # Use DNI to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # Initialize the RealESRGANer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id
    )

    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print(f'Testing {idx}: {imgname}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_mode = 'RGBA' if len(img.shape) == 3 and img.shape[2] == 4 else None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print(f'Error: {error}')
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            continue

        extension = extension[1:] if args.ext == 'auto' else args.ext
        if img_mode == 'RGBA':
            extension = 'png'
        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}' if args.suffix else f'{imgname}.{extension}')
        cv2.imwrite(save_path, output)

if __name__ == '__main__':
    main()
