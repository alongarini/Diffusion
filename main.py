import csv
import logging
import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from skimage.metrics import structural_similarity as ssim

from src.config.base_config import ModelConfig
from src.data.dataloader import prepare_data
from src.metrics.image_metrics import ImageQualityMetric
from src.metrics.performance_metrics import PerformanceMetric
from src.metrics.segmentation_metrics import AssertivenessMetric
from src.models.cnn import RecursiveCNN
from src.models.diffusion import DenoiseNetwork, DiffusionModel
from src.models.gan import Discriminator, GANTrainer, Generator
from src.utils.helpers import (export_metrics_csv, get_true_labels,
                               load_checkpoint, plot_confusion_matrix,
                               plot_histograms, plot_metrics, plot_samples,
                               save_checkpoint, setup_logging, train_model,
                               validate_model)
from src.utils.postprocessing import apply_post_processing

# Global configuration (could be moved to a config file or argparse)
RESULTS_DIR = "./resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = [
    {"model": "GAN", "mask": False},
    {"model": "GAN", "mask": True},
    {"model": "Diffusion", "mask": False},
    {"model": "Diffusion", "mask": True},
]


def prepare_dataloaders(config: ModelConfig, with_mask: bool) -> Tuple[Any, Any, Any]:
    """
    Prepara os dataloaders de treino, validação e teste.

    Args:
        config (ModelConfig): Objeto de configuração do pipeline.
        with_mask (bool): Se True, utiliza máscaras no dataset.

    Returns:
        Tuple: (train_loader, val_loader, test_loader)
    """
    return prepare_data(
        config.training.data_dir,
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


def evaluate_images(originals, generated, config) -> Dict[str, float]:
    """
    Avalia imagens geradas usando FID, SSIM e PSNR.

    Args:
        originals: Tensor de imagens originais.
        generated: Tensor de imagens geradas.
        config (ModelConfig): Objeto de configuração.

    Returns:
        dict: Métricas calculadas (fid, ssim, psnr).
    """
    orig_uint8 = to_uint8_tensor(originals)
    gen_uint8 = to_uint8_tensor(generated)
    metric = ImageQualityMetric('comparison')
    fid = metric.calculate_fid(orig_uint8, gen_uint8, device=config.device)
    ssim_vals, psnr_vals = zip(*[
        metric.calculate_ssim_psnr(
            o.permute(1, 2, 0).cpu().numpy(),
            g.permute(1, 2, 0).cpu().numpy()
        )
        for o, g in zip(orig_uint8, gen_uint8)
    ])
    ssim = sum(ssim_vals) / len(ssim_vals)
    psnr = sum(psnr_vals) / len(psnr_vals)
    return {"fid": fid, "ssim": ssim, "psnr": psnr}


def main(config_path: str = "config/model_config.yaml"):
    """
    Pipeline principal para comparar modelos GAN e Diffusion em dados de imagem.

    Args:
        config_path (str): Caminho para o arquivo de configuração YAML.
    """
    logger = setup_logging()
    logger.info('Início da execução do pipeline de comparação de modelos.')

    # Load configurations
    logger.info(f'Carregando configuração do arquivo: {config_path}')
    config = ModelConfig(config_path)
    try:
        config.validate()
        logger.info('Configuração validada com sucesso.')
    except Exception as e:
        logger.error(f'Erro de validação da configuração: {e}', exc_info=True)
        raise
    logger.info(f"Configuração carregada: {config}")

    try:
        logger.info('Inicializando modelos...')
        diffusion_model, gan_trainer, cnn_model = initialize_models(config)
        logger.info('Modelos inicializados com sucesso.')
        results = {}
        perf_results = {}
        perf_metric = PerformanceMetric('performance')
        for scenario in SCENARIOS:
            logger.info(
                f"Iniciando cenário: {scenario['model']} {'com máscara' if scenario['mask'] else 'sem máscara'}")
            train_loader, val_loader, test_loader = prepare_dataloaders(
                config, with_mask=scenario["mask"])
            # Training
            if scenario["model"] == "Diffusion":
                optimizer_diff = torch.optim.Adam(
                    diffusion_model.denoise_network.parameters(),
                    lr=config.diffusion.learning_rate
                )
                logger.info(
                    f"Treinando modelo Diffusion por {config.diffusion.num_epochs} épocas...")
                _, train_time = perf_metric.measure_time(
                    train_model, diffusion_model, train_loader, optimizer_diff,
                    config.device, config.diffusion.num_epochs, 'diffusion'
                )
                logger.info('Treinamento Diffusion finalizado.')
                val_loss = validate_model(
                    diffusion_model, val_loader, config.device, model_type='diffusion')
                logger.info(f"Validação Diffusion: loss médio = {val_loss}")
                save_checkpoint(
                    diffusion_model.denoise_network,
                    f"{config.training.save_dir}/diffusion_{'mask' if scenario['mask'] else 'no_mask'}.pth"
                )
                logger.info('Checkpoint do modelo Diffusion salvo.')
                originals, generated = generate_images(
                    test_loader, "Diffusion", diffusion_model, gan_trainer, config)
            else:
                logger.info(
                    f"Treinando modelo GAN por {config.gan.num_epochs} épocas...")
                _, train_time = perf_metric.measure_time(
                    train_model, gan_trainer, train_loader, None,
                    config.device, config.gan.num_epochs, 'gan'
                )
                logger.info('Treinamento GAN finalizado.')
                val_loss = validate_model(
                    gan_trainer, val_loader, config.device, model_type='gan')
                logger.info(f"Validação GAN: loss médio = {val_loss}")
                save_checkpoint(
                    gan_trainer.G,
                    f"{config.training.save_dir}/ganG_{'mask' if scenario['mask'] else 'no_mask'}.pth"
                )
                save_checkpoint(
                    gan_trainer.D,
                    f"{config.training.save_dir}/ganD_{'mask' if scenario['mask'] else 'no_mask'}.pth"
                )
                logger.info('Checkpoints do modelo GAN salvos.')
                originals, generated = generate_images(
                    test_loader, "GAN", diffusion_model, gan_trainer, config)

            # CNN training
            logger.info(
                f"Treinando modelo CNN por {config.cnn.num_epochs} épocas...")
            optimizer_cnn = torch.optim.Adam(
                cnn_model.parameters(), lr=config.cnn.learning_rate)
            _, train_time_cnn = perf_metric.measure_time(
                train_model, cnn_model, train_loader, optimizer_cnn,
                config.device, config.cnn.num_epochs, 'cnn'
            )
            logger.info('Treinamento CNN finalizado.')
            val_loss_cnn = validate_model(
                cnn_model, val_loader, config.device, model_type='cnn')
            logger.info(f"Validação CNN: loss médio = {val_loss_cnn}")
            save_checkpoint(
                cnn_model,
                f"{config.training.save_dir}/cnn_{scenario['model']}_{'mask' if scenario['mask'] else 'no_mask'}.pth"
            )
            logger.info('Checkpoint do modelo CNN salvo.')
            processed_generated = apply_post_processing(generated)
            class_cnn = classify_images(processed_generated, cnn_model, config)
            metrics = evaluate_images(originals, processed_generated, config)
            key = f"{scenario['model']}_{'mask' if scenario['mask'] else 'no_mask'}"
            results[key] = metrics
            logger.info(f"Métricas para {key}: {metrics}")
            # Confusion matrix and visualizations
            y_true = get_true_labels(test_loader)
            y_pred = torch.argmax(class_cnn, dim=1).cpu().numpy()
            plot_confusion_matrix(
                y_true, y_pred,
                labels=["Class 0", "Class 1"],
                filename=f"{config.training.save_dir}/confusion_matrix_{key}.png"
            )
            logger.info(
                f"Matriz de confusão salva em {config.training.save_dir}/confusion_matrix_{key}.png")
            plot_histograms(originals, processed_generated,
                            key, config.training.save_dir)
            plot_samples(originals, processed_generated,
                         key, config.training.save_dir)
            logger.info(f"Histogramas e amostras salvos para {key}")
            # Performance metrics
            mem = perf_metric.measure_memory()
            mem_gpu = perf_metric.measure_memory_gpu()
            total_time = train_time + train_time_cnn
            cost_img = perf_metric.cost_per_image(
                total_time, len(test_loader.dataset))
            logger.info(
                f"Performance: tempo total={total_time:.2f}s, memória={mem:.2f}MB, GPU memória={mem_gpu}, custo por imagem={cost_img}")
            # FLOPs (example for CNN)
            flops, params = None, None
            if hasattr(cnn_model, 'fc'):
                input_res = (3, config.image_size, config.image_size)
                flops, params = perf_metric.measure_flops(cnn_model, input_res)
                logger.info(f"CNN FLOPs: {flops}, parâmetros: {params}")
            # Save performance metrics
            with open(f"{config.training.save_dir}/performance_{key}.txt", "w") as f:
                for r in perf_metric.get_results():
                    f.write(str(r) + "\n")
            perf_results[key] = {
                'total_time': total_time,
                'memory_MB': mem,
                'gpu_memory_MB': mem_gpu,
                'cost_per_image': cost_img,
                'flops': flops,
                'params': params
            }
            logger.info(f"Cenário {key} finalizado.")
        # Plot and export results
        for metric in ["fid", "ssim", "psnr"]:
            plot_metrics(results, metric, config.training.save_dir)
        logger.info("Relatório final de comparação:")
        for key, metrics in results.items():
            logger.info(f"{key}: {metrics}")
        best = min(results.items(), key=lambda x: x[1]["fid"])
        logger.info(f"Melhor modelo (menor FID): {best[0]}")
        export_metrics_csv(results, perf_results,
                           f"{config.training.save_dir}/comparative_metrics.csv")
        logger.info(
            f"Métricas numéricas exportadas para {config.training.save_dir}/comparative_metrics.csv")
    except Exception as e:
        logger.error(
            f"Erro durante a execução do pipeline: {e}", exc_info=True)
    logger.info('Fim da execução do pipeline.')


if __name__ == "__main__":
    main()
