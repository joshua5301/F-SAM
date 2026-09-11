# PG-SAM launcher. The recipe below matches run.sh; per-arm flags pass through:
#   bash run_pgsam.sh                                   # SGD floor
#   bash run_pgsam.sh --gates channel --gate-rho 0.05   # PG-SAM(ch)
#   bash run_pgsam.sh --perturb bn --rho 0.5            # SAM-ON
#   bash run_pgsam.sh --perturb all --rho 0.2           # SAM
device=0
seed=1
datasets=CIFAR100
model=${MODEL:-resnet18}   # resnet18 VGG16BN WideResNet28x10  (override: MODEL=... bash run_pgsam.sh ...)
schedule=cosine
wd=0.001
epoch=200
bz=128
lr=0.05

TAG=$(echo "${*:-sgd}" | sed 's/--//g; s/ /_/g; s/,/+/g')
DST=results/PGSAM/$datasets/$model/${TAG}_seed$seed

CUDA_VISIBLE_DEVICES=$device python -u train.py --datasets $datasets \
        --arch=$model --epochs=$epoch --wd=$wd --randomseed $seed --lr $lr --optimizer PGSAM \
        --save-dir=$DST/checkpoints --log-dir=$DST -p 200 --schedule $schedule -b $bz \
        --cutout "$@"
