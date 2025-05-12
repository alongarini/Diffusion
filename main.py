# fmt: off
# isort: skip

# Caminho absoluto para o diretório MaskTheFace
# masktheface_path = os.path.join(os.getcwd(), 'MaskTheFace')
# sys.path.append(masktheface_path)
import csv
import logging
import os
import sys
from typing import Any, Dict, Tuple

# isort: skip
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from scipy.stats import pearsonr
# Agora você pode importar os módulos do MaskTheFace
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, precision_recall_curve

# from MaskTheFace import mask_the_face
from src.config.base_config import ModelConfig
from src.data.dataloader import prepare_data
from src.metrics.image_metrics import ImageQualityMetric
from src.metrics.performance_metrics import PerformanceMetric
from src.metrics.segmentation_metrics import AssertivenessMetric
from src.models.cnn import RecursiveCNN
from src.models.diffusion import DenoiseNetwork, DiffusionModel
from src.models.gan import Discriminator, GANTrainer, Generator
from src.pipeline.results import save_checkpoint, save_metrics
from src.pipeline.runner import prepare_dataloaders
from src.pipeline.scenario import generate_images, initialize_models
from src.pipeline.tuning import run_hyperparameter_tuning
from src.pipeline.visualization import plot_comparative_graph
from src.utils.helpers import (export_metrics_csv, get_true_labels,
                               load_checkpoint, plot_confusion_matrix,
                               plot_histograms, plot_metrics, plot_samples,
                               save_checkpoint, setup_logging, train_model,
                               validate_model)
from src.utils.postprocessing import apply_post_processing

# Global configuration (could be moved to a config file or argparse)
RESULTS_DIR = "./resultados"
RECONSTRUCTED_DIR = os.path.join(RESULTS_DIR, "reconstruidas")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)

SCENARIOS = [
    {"model": "GAN", "mask": False},
    {"model": "GAN", "mask": True},
    {"model": "Diffusion", "mask": False},
    {"model": "Diffusion", "mask": True},
]


def prepare_dataloaders(config: ModelConfig, with_mask: bool) -> Tuple[Any, Any, Any]:
    """
    Prepara os dataloaders de treino, validação e teste.
    Para with_mask=True, utiliza imagens mascaradas (config.training.mask_dir).
    Para with_mask=False, utiliza imagens originais (config.training.data_dir).
    A correspondência entre imagens originais e mascaradas é garantida pelo nome do arquivo.

    Args:
        config (ModelConfig): Objeto de configuração do pipeline.
        with_mask (bool): Se True, utiliza imagens mascaradas; caso contrário, imagens originais.

    Returns:
        Tuple: (train_loader, val_loader, test_loader)
    """
    data_dir = config.training.mask_dir if with_mask else config.training.data_dir
    return prepare_data(
        data_dir,
        config.batch_size,
        config.image_size,
        with_mask=with_mask
    )


def initialize_models(config: ModelConfig) -> Tuple[DiffusionModel, GANTrainer, RecursiveCNN]:
    """
    Inicializa todos os modelos necessários usando a configuração fornecida.

    Args:
        config (ModelConfig): Instância contendo todas as configurações dos modelos.

    Returns:
        Tuple: (diffusion_model, gan_trainer, cnn_model)
    """
    # Initialize Diffusion model
    denoise_net = DenoiseNetwork(T=config.diffusion.T)
    diffusion_model = DiffusionModel(
        denoise_network=denoise_net,
        T=config.diffusion.T,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        device=config.diffusion.device
    )

    # Initialize GAN
    generator = Generator(noise_dim=config.gan.noise_dim)
    discriminator = Discriminator()
    gan_trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        noise_dim=config.gan.noise_dim,
        lr_D=config.gan.lr_discriminator,
        lr_G=config.gan.lr_generator,
        batch_size=config.gan.batch_size,
        device=config.gan.device
    )

    # Initialize CNN
    cnn_model = RecursiveCNN(
        in_channels=config.cnn.in_channels,
        out_channels=config.cnn.out_channels,
        kernel_size=config.cnn.kernel_size,
        num_iterations=config.cnn.num_iterations,
        num_classes=config.cnn.num_classes
    ).to(config.cnn.device)
    return diffusion_model, gan_trainer, cnn_model


def generate_images(test_loader, model_type, diffusion_model, gan_trainer, config):
    """
    Gera imagens usando o modelo especificado (GAN ou Diffusion).

    Args:
        test_loader: DataLoader de teste.
        model_type (str): Tipo do modelo ('GAN' ou 'Diffusion').
        diffusion_model: Instância do modelo Diffusion.
        gan_trainer: Instância do treinador GAN.
        config (ModelConfig): Objeto de configuração.

    Returns:
        Tuple: (originals, generated) - tensores de imagens originais e geradas.
    """
    originals = []
    generated = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(config.device)
            batch_size = images.size(0)
            originals.append(images.cpu())
            generated_images = None
            if model_type == "Diffusion":
                t_value = int(config.diffusion.T // 2)
                t = torch.ones(
                    batch_size, device=config.device).long() * t_value
                generated_images = images + 0.1 * torch.randn_like(images)
            elif model_type == "GAN":
                noise = torch.randn(
                    batch_size, config.gan.noise_dim, 1, 1, device=config.device)
                generated_images = gan_trainer.generator(noise)
            generated.append(generated_images.cpu())
    originals = torch.cat(originals, dim=0)
    generated = torch.cat(generated, dim=0)
    return originals, generated


def classify_images(images, cnn_model, config):
    """
    Classifica imagens usando o modelo CNN.

    Args:
        images: Tensor de imagens a serem classificadas.
        cnn_model: Instância do modelo CNN.
        config (ModelConfig): Objeto de configuração.

    Returns:
        Tensor: Predições do modelo CNN.
    """
    cnn_model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(images), config.batch_size):
            batch = images[i:i+config.batch_size].to(config.device)
            preds, _ = cnn_model(batch)
            results.append(preds.cpu())
    return torch.cat(results, dim=0)


def to_uint8_tensor(imgs):
    """
    Converte imagens para tensor uint8.

    Args:
        imgs: Tensor de imagens normalizadas.

    Returns:
        Tensor: Imagens no formato uint8.
    """
    imgs = imgs.clone()
    if imgs.min() < 0:
        imgs = (imgs + 1) / 2
    imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
    return imgs


def calculate_map(real_masks, generated_masks):
    """
    Calcula o Mean Average Precision (MAP) entre máscaras reais e geradas.
    Args:
        real_masks: lista/array de máscaras reais
        generated_masks: lista/array de máscaras geradas
    Returns:
        MAP médio
    """
    map_values = []
    for real, gen in zip(real_masks, generated_masks):
        precision, recall, _ = precision_recall_curve(real.flatten(), gen.flatten())
        map_values.append(auc(recall, precision))
    return float(np.mean(map_values)) if map_values else None


def evaluate_images(originals, generated, config, real_masks=None, generated_masks=None) -> dict:
    """
    Avalia imagens geradas comparando-as com as imagens originais utilizando métricas de qualidade de imagem.
    Agora inclui também MSE, MAE, MAPE e Pearson.
    """
    orig_uint8 = to_uint8_tensor(originals)
    gen_uint8 = to_uint8_tensor(generated)
    metric = ImageQualityMetric('comparison')
    seg_metric = AssertivenessMetric('assertiveness')
    fid = metric.calculate_fid(orig_uint8, gen_uint8, device=config.device)
    ssim_vals, psnr_vals, mse_vals, mae_vals, mape_vals, pearson_vals = [], [], [], [], [], []
    for o, g in zip(orig_uint8, gen_uint8):
        o_np = o.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        g_np = g.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        ssim, psnr = metric.calculate_ssim_psnr(o_np, g_np)
        ssim_vals.append(ssim)
        psnr_vals.append(psnr)
        mse = seg_metric.mse(o_np.flatten(), g_np.flatten())
        mae = seg_metric.mae(o_np.flatten(), g_np.flatten())
        mape = seg_metric.mape(o_np.flatten(), g_np.flatten())
        # Pearson: se as imagens forem constantes, retorna nan, então tratamos
        try:
            pearson, _ = pearsonr(o_np.flatten(), g_np.flatten())
        except Exception:
            pearson = np.nan
        mse_vals.append(mse)
        mae_vals.append(mae)
        mape_vals.append(mape)
        pearson_vals.append(pearson)
    # Cálculo do MAP se máscaras disponíveis
    map_score = None
    if real_masks is not None and generated_masks is not None:
        map_score = calculate_map(real_masks, generated_masks)
    return {
        "fid": fid,
        "ssim": float(np.mean(ssim_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "mse": float(np.mean(mse_vals)),
        "mae": float(np.mean(mae_vals)),
        "mape": float(np.mean(mape_vals)),
        "pearson": float(np.nanmean(pearson_vals)),
        "map": map_score,
    }


def plot_comparativo(original, gan, diffusion, ssim_gan, psnr_gan, ssim_diff, psnr_diff, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title(f"GAN\nSSIM: {ssim_gan:.4f} | PSNR: {psnr_gan:.2f}")
    plt.imshow(gan)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title(f"Diffusion\nSSIM: {ssim_diff:.4f} | PSNR: {psnr_diff:.2f}")
    plt.imshow(diffusion)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def evaluate_images_per_image(originals, generated, config, img_names, save_path, real_masks=None, generated_masks=None):
    """
    Avalia cada par de imagens (original x reconstruída) individualmente e salva as métricas em um CSV.
    Agora inclui também MSE, MAE, MAPE e Pearson.
    """
    orig_uint8 = to_uint8_tensor(originals)
    gen_uint8 = to_uint8_tensor(generated)
    metric = ImageQualityMetric('comparison')
    seg_metric = AssertivenessMetric('assertiveness')
    rows = []
    for i, (o, g) in enumerate(zip(orig_uint8, gen_uint8)):
        o_np = o.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        g_np = g.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        ssim, psnr = metric.calculate_ssim_psnr(o_np, g_np)
        mse = seg_metric.mse(o_np.flatten(), g_np.flatten())
        mae = seg_metric.mae(o_np.flatten(), g_np.flatten())
        mape = seg_metric.mape(o_np.flatten(), g_np.flatten())
        try:
            pearson, _ = pearsonr(o_np.flatten(), g_np.flatten())
        except Exception:
            pearson = np.nan
        # MAP por imagem se máscaras disponíveis
        map_score = None
        if real_masks is not None and generated_masks is not None:
            precision, recall, _ = precision_recall_curve(real_masks[i].flatten(), generated_masks[i].flatten())
            map_score = auc(recall, precision)
        row = {
            'img_name': img_names[i],
            'ssim': ssim,
            'psnr': psnr,
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'pearson': pearson,
            'map': map_score
        }
        rows.append(row)
    # Salva o CSV
    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ['img_name', 'ssim', 'psnr', 'mse', 'mae', 'mape', 'pearson', 'map']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_performance_metrics(perf_metric, key, save_dir):
    """
    Salva as métricas de performance em um arquivo formatado, incluindo descrições explicativas das métricas.
    Args:
        perf_metric: Objeto PerformanceMetric
        key: Nome do cenário/modelo
        save_dir: Diretório para salvar o arquivo
    """
    with open(f"{save_dir}/performance_{key}.txt", "w") as f:
        # Cabeçalho
        f.write("# Métricas de Performance\n\n")
        # Tempo de Execução
        f.write("## Tempo de Execução (em segundos)\n")
        f.write("Tempo gasto em cada operação do modelo durante o processamento.\n")
        for r in perf_metric.get_results():
            if 'execution_time' in r:
                f.write(f"Tempo da operação: {r['execution_time']:.2f}s\n")
        f.write("\n")
        # Uso de Memória
        f.write("## Uso de Memória\n")
        f.write("Uso de memória RAM (em MB) durante a execução do modelo.\n")
        for r in perf_metric.get_results():
            if 'memory_MB' in r:
                f.write(f"RAM: {r['memory_MB']:.2f} MB\n")
            if 'gpu_memory_MB' in r:
                f.write(f"GPU: {r['gpu_memory_MB']:.2f} MB\n")
        f.write("\n")
        # Custo por Imagem
        f.write("## Custo Computacional\n")
        f.write("Tempo médio de processamento por imagem (em segundos por imagem).\n")
        for r in perf_metric.get_results():
            if 'cost_per_image' in r:
                f.write(f"Custo por imagem: {r['cost_per_image']:.3f} segundos\n")
        f.write("\n")
        # Complexidade do Modelo
        f.write("## Complexidade do Modelo\n")
        f.write("FLOPs: Número de operações de ponto flutuante realizadas pelo modelo (não calculado neste experimento).\n")
        f.write("Parâmetros: Quantidade total de parâmetros do modelo (não calculado neste experimento).\n")
        for r in perf_metric.get_results():
            if 'FLOPs' in r or 'params' in r:
                f.write(f"FLOPs: {r.get('FLOPs', 'N/A')}\n")
                f.write(f"Parâmetros: {r.get('params', 'N/A')}\n")


def main(config_path: str = "config/model_config.yaml"):
    """
    Função principal do pipeline: orquestra o fluxo de execução utilizando os módulos do pipeline.
    """
    logger = setup_logging()
    logger.info('Início da execução do pipeline de comparação de modelos.')

    # Carregar configuração
    config = ModelConfig(config_path)

    # Tuning de hiperparâmetros (se ativado)
    if config.hyperparameter_tuning.get('enabled', False):
        logger.info('Ajuste automático de hiperparâmetros ATIVADO.')
        config = run_hyperparameter_tuning(config)
    else:
        logger.info('Ajuste automático de hiperparâmetros DESATIVADO. Usando valores do YAML.')

    # Validar configuração
    try:
        config.validate()
        logger.info('Configuração validada com sucesso.')
    except Exception as e:
        logger.error(f'Erro de validação da configuração: {e}', exc_info=True)
        raise
    logger.info(f"Configuração carregada: {config}")

    # Inicializar modelos
    diffusion_model, gan_trainer, cnn_model = initialize_models(config)
    logger.info('Modelos inicializados com sucesso.')

    # Loop de cenários
    for scenario in SCENARIOS:
        logger.info(f"Iniciando cenário: {scenario['model']} {'com máscara' if scenario['mask'] else 'sem máscara'}")
        train_loader, val_loader, test_loader = prepare_dataloaders(config, with_mask=scenario["mask"])
        # Aqui você pode chamar funções de treinamento, avaliação, geração de imagens, etc.
        # Exemplo:
        # originals, generated = generate_images(test_loader, scenario["model"], diffusion_model, gan_trainer, config)
        # ...
        # Salvar métricas, checkpoints, gráficos, etc.
        # save_metrics(...)
        # save_checkpoint(...)
        # plot_metrics(...)
        # plot_comparative_graph(...)
        pass  # TODO: Integrar todo o fluxo detalhado usando os módulos do pipeline

    logger.info('Fim da execução do pipeline.')


if __name__ == "__main__":
    main()


# TODO: Executar o código com o mascaramento das imagens e comparar com o código sem o mascaramento.
# Salvar todos os resultados em um arquivo csv e imagens no stdout OU notebook