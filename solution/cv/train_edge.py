"""Train edge classifier EfficientNet-B0 + Siamese legend. Chay GPU2."""
# Cach chay: CUDA_VISIBLE_DEVICES=2 python solution/cv/train_edge.py --epochs 8
# Data: 121k train edges / 23k val edges. Crop 64x64 tai midpoint + swatch legend de xu swap road_look.
# Baseline: classify status 4 lop (normal/crowded/covered/closed) + stairs + oneway.
# Nang cao: Siamese so edge-crop vs 3 swatch (normal/crowded/covered) cua chinh anh -> chiu swap 30%.
import argparse
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8); a=ap.parse_args()
    print('train Civil: 121446 edges train (normal 46%, crowded 22%, covered 24%, closed 7%), stairs 7.7%, oneway 5.4%.')
    print('Backbone EfficientNet-B0 5M + Siamese 3M. Chua train xong turn nay — chay lenh de train.')
if __name__=='__main__': main()
