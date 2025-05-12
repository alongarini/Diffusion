# Projeto: Comparativo entre Modelos GANs e Diffusion

## Objetivo

O objetivo deste projeto é comparar o desempenho de modelos GANs e Diffusion na geração de imagens, utilizando um fluxo robusto de machine learning com conjuntos de treino, validação e teste. Após a geração, as imagens passam por um classificador CNN e são avaliadas por um sistema completo de métricas de qualidade, assertividade e performance.

---

## Estrutura dos Módulos Principais

```
src/
  pipeline/
    runner.py           # Orquestração do pipeline principal
    scenario.py         # Execução de cenários (treinamento, avaliação, salvamento)
    tuning.py           # Lógica de tuning de hiperparâmetros
    visualization.py    # Funções de visualização e geração de gráficos
    results.py          # Salvamento de métricas, checkpoints e resultados
  models/
  data/
  metrics/
  utils/
```

### Descrição dos módulos do pipeline
- `runner.py`: Função principal do pipeline, responsável por orquestrar o fluxo.
- `scenario.py`: Inicialização de modelos, geração de imagens e execução de cenários.
- `tuning.py`: Ajuste automático de hiperparâmetros.
- `visualization.py`: Geração de gráficos e visualizações.
- `results.py`: Salvamento de métricas e checkpoints.

---

## Fluxo de Execução Simplificado

O arquivo `main.py` agora apenas orquestra o pipeline, delegando responsabilidades para os módulos do diretório `src/pipeline/`.

Exemplo de uso:
```python
from src.pipeline.runner import run_pipeline

if __name__ == "__main__":
    run_pipeline("config/model_config.yaml")
```

---

## Estrutura do Pipeline

O pipeline é dividido em três etapas principais: treinamento, validação e teste, seguindo as melhores práticas de machine learning.

### 1. Preparação dos Dados
```python
train_loader, val_loader, test_loader = prepare_dataloaders(
    batch_size=config.batch_size,
    image_size=config.image_size,
    with_mask=scenario["mask"]
)
```
- Divisão em conjuntos de treino, validação e teste
- Uso de DataLoaders para gerenciamento eficiente de memória
- Configuração flexível de batch size e tamanho de imagem

### 2. Treinamento dos Modelos

#### a. Modelo Diffusion
```python
optimizer_diff = torch.optim.Adam(diffusion_model.denoise_network.parameters(), lr=2e-4)
_, train_time = perf_metric.measure_time(
    train_model, diffusion_model, train_loader, optimizer_diff, 
    device, num_epochs, 'diffusion'
)
```

#### b. Modelo GAN
```python
_, train_time = perf_metric.measure_time(
    train_model, gan_trainer, train_loader, None, 
    device, num_epochs, 'gan'
)
```

#### c. Classificador CNN
```python
optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
_, train_time_cnn = perf_metric.measure_time(
    train_model, cnn_model, train_loader, optimizer_cnn, 
    device, num_epochs, 'cnn'
)
```

### 3. Validação dos Modelos

#### a. Modelo Diffusion
```python
val_loss = validate_model(diffusion_model, val_loader, device, model_type='diffusion')
logger.info(f"Diffusion validation: mean loss = {val_loss}")
```

#### b. Modelo GAN
```python
val_loss = validate_model(gan_trainer, val_loader, device, model_type='gan')
logger.info(f"GAN validation: mean loss = {val_loss}")
```

#### c. Classificador CNN
```python
val_loss_cnn = validate_model(cnn_model, val_loader, device, model_type='cnn')
logger.info(f"CNN validation: mean loss = {val_loss_cnn}")
```

### 4. Teste e Avaliação

#### a. Geração de Imagens
```python
originals, generated = generate_images(test_loader, model_type, diffusion_model, gan_trainer)
processed_generated = apply_post_processing(generated)
```

#### b. Classificação das Imagens Geradas
```python
class_cnn = classify_images(processed_generated, cnn_model)
y_true = get_true_labels(test_loader)
y_pred = torch.argmax(class_cnn, dim=1).cpu().numpy()
```

#### c. Avaliação de Métricas
```python
metrics = evaluate_images(originals, processed_generated, device)
results[key] = metrics
logger.info(f"Metrics for {key}: {metrics}")
```

#### d. Visualizações e Métricas de Performance
```python
# Visualizações
plot_confusion_matrix(y_true, y_pred, labels=["Class 0", "Class 1"])
plot_histograms(originals, processed_generated, key)
plot_samples(originals, processed_generated, key)

# Métricas de Performance
mem = perf_metric.measure_memory()
mem_gpu = perf_metric.measure_memory_gpu()
total_time = train_time + train_time_cnn
cost_img = perf_metric.cost_per_image(total_time, len(test_loader.dataset))
```

### 5. Cálculo e Exportação de Métricas
   - **Qualidade de Imagem:** SSIM, PSNR, FID, LPIPS, histogramas RGB/intensidade.
   - **Assertividade:** Accuracy, precision, recall, f1, matriz de confusão, MAE, MSE, RMSE, MAPE.
   - **Performance:** Tempo de execução, uso de memória RAM/GPU, FLOPs, custo por imagem/batch.
   - Todas as métricas são salvas em arquivos CSV (numéricas) e PNG (visuais).

5. **Visualizações e Relatórios**
   - Gráficos comparativos, histogramas, matrizes de confusão e amostras de imagens são exportados automaticamente.
   - Relatórios de performance e tabelas de comparação são gerados para cada cenário.

6. **Exportação**
   - Todos os resultados (numéricos e visuais) são salvos em diretórios organizados, prontos para análise e inclusão em relatórios/dissertação.

---

## Logging

O sistema de logging monitora todo o processamento do pipeline. Os logs são salvos em `./logs/pipeline.log` e exibidos no console.

- Níveis: INFO, DEBUG, WARNING, ERROR.
- Rotação automática de arquivos.
- Formato com timestamp, nível e contexto.

Exemplo de uso:
```python
from src.utils.helpers import setup_logging
logger = setup_logging(level=logging.INFO)
logger.info('Mensagem informativa')
```

---

## Sistema de Métricas

O sistema de métricas está organizado em três categorias principais:

- **Qualidade da Imagem:** `ImageQualityMetric`
- **Assertividade:** `AssertivenessMetric`
- **Performance:** `PerformanceMetric`

Cada categoria possui uma classe específica que herda de uma classe base abstrata (`MetricBase`).

Exemplo de uso:
```python
from src.metrics.image_metrics import ImageQualityMetric
from src.metrics.segmentation_metrics import AssertivenessMetric
from src.metrics.performance_metrics import PerformanceMetric

# Qualidade de imagem
img_metric = ImageQualityMetric('ssim_psnr')
ssim, psnr = img_metric.calculate_ssim_psnr(img1, img2)

# Assertividade
assert_metric = AssertivenessMetric('f1')
result = assert_metric.calculate_segmentation_metrics(mask_real, mask_pred)

# Performance
perf_metric = PerformanceMetric('tempo')
_, tempo = perf_metric.measure_time(funcao_a_medida, *args)
mem = perf_metric.measure_memory()
```

---

## Como executar este projeto:

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/projeto-gan-diffusion.git
cd projeto-gan-diffusion
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure o projeto:
```bash
# Os parâmetros devem ser informados no arquivo:
/config/model_config.yaml
# Consulte o próprio arquivo para explicação de cada parâmetro.
```

4. Execute o projeto:
```bash
python main.py
```

---

## Padronização de Docstrings

Todas as funções e módulos do pipeline seguem o padrão abaixo para docstrings:

```python
def minha_funcao(param1: int, param2: str) -> bool:
    """
    Descrição sucinta da função.

    Args:
        param1 (int): Descrição do parâmetro 1.
        param2 (str): Descrição do parâmetro 2.

    Returns:
        bool: Descrição do valor de retorno.
    """
    # implementação
```

---

## Exportação de Resultados

- Todos os resultados (numéricos e visuais) são salvos em diretórios organizados, prontos para análise e inclusão em relatórios/dissertação.
- Gráficos comparativos, histogramas, matrizes de confusão e amostras de imagens são exportados automaticamente.
- Relatórios de performance e tabelas de comparação são gerados para cada cenário.

---

## Observações
- O pipeline foi refatorado para facilitar manutenção, testes e extensibilidade.
- Consulte os módulos em `src/pipeline/` para detalhes de implementação e exemplos de uso avançado.

---

## Próximos Passos Sugeridos

- Refino de visualizações (gráficos de radar, linha, normalização, etc.)
- Testes finais com o dataset completo
- Revisão e complementação da documentação
- Preparação de material visual para apresentação e defesa

---

Este projeto está pronto para ser utilizado em produção e para servir de base para a defesa de mestrado, garantindo reprodutibilidade, análise visual e numérica, e documentação adequada.