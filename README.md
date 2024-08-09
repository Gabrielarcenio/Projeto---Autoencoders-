# Fashion MNIST Denoising and Classification

Este projeto usa autoencoders para remover ruído das imagens do dataset Fashion MNIST e uma MLP (Multi-Layer Perceptron) para classificação.

## Requisitos

As bibliotecas necessárias para rodar este projeto estão listadas em `requirements.txt`.

## Estrutura do Projeto

- `fashion_mnist_denoising_classification.ipynb`: Notebook principal contendo o código para treinamento e avaliação dos modelos.
- `src/`: Contém os scripts Python organizados:
  - `autoencoder_model.py`: Definição do modelo de autoencoder.
  - `classifier_model.py`: Definição da MLP para classificação.
  - `data_loader.py`: Carregamento e pré-processamento dos dados do Fashion MNIST.
  - `utils.py`: Funções auxiliares.

## Como Executar

1. Clone este repositório.
2. Instale as dependências usando `pip install -r requirements.txt`.
3. Abra o notebook `fashion_mnist_denoising_classification.ipynb` e execute as células para treinar e avaliar os modelos.

## Autor

Gabriel Arcenio Rahal Marostica
