import os
import shutil
import subprocess

def run(cmd, step_name):
    print(f"\n🔧 Running: {step_name}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {step_name} failed! Exiting.\n")
        exit(1)

# CLEAR MODELS AND PLOTS
shutil.rmtree("models/cifar10", ignore_errors=True)
shutil.rmtree("plots/cifar10", ignore_errors=True)

gpu = "0"
dataset = "cifar10"
epochs = "50"
final_epoch = "49"
taus = "2.0"
alphas = "1.0"
model_baseline = "deep_cnn"
model_regularized = "deep_cnn_regularized"


run(f"""
python experiment_main_regularized.py \
    --dataset={dataset} --gpu={gpu} --epochs={epochs} \
    --model_names={model_baseline},{model_regularized} \
    --with_regularization --taus={taus} --alphas={alphas} \
    --sigmoid_approx --probabilities \
    --betas=none --gammas=none
""", "Step 1: Train models")

run(f"""
python experiment-adversarial.py \
    --dataset={dataset} --gpu={gpu} --model_names={model_baseline},{model_regularized} \
    --epochs={final_epoch} --taus={taus} --alphas={alphas} \
    --with_regularization --sigmoid_approx --probabilities
""", "Step 2: Run adversarial attacks")

for model in [model_baseline, model_regularized]:
    run(f"""
    python experiment-adversarial-plot-only.py \
        --dataset={dataset} --gpu={gpu} --model_name={model} \
        --epochs={final_epoch} --taus={taus} --alphas={alphas} \
        --with_regularization --sigmoid_approx --probabilities
    """, f"Step 3: Plot robustness bias for {model}")

print("\n✅ All steps completed successfully!")
