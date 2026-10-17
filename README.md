# Protótipo VSLM

Modelo experimental de previsão do próximo token, inspirado na decomposição Attencioner, Modelo Principal e De-vectorizer. Cada token usa um vetor de **256 dimensões**, representado por uma fatoração compartilhada de rank **48** entre Attencioner e De-vectorizer. O modelo usa **todos os IDs do tokenizer configurado**; no GPT-2, são 50.257 tokens originais mais o token UNK adicionado pelo treino. Nenhum ID é removido com base na frequência. A fatoração representa a tabela como `tamanho_do_tokenizer × 48` e `48 × 256`, reduzindo parâmetros em relação a uma tabela densa `tamanho_do_tokenizer × 256`. `train.py` imprime a contagem real de parâmetros, separando os fatores compartilhados. Os vetores de saída do Attencioner usam fake quantization para **4 bits por dimensão**; os pesos continuam em float32.

O Attencioner recebe o bloco inteiro `[batch, sequência]` em uma chamada. No treino, o Modelo Principal também processa todas as posições em paralelo: convoluções causais depthwise de janela 128 misturam o contexto passado sem permitir acesso a tokens futuros. O loop de treino percorre as camadas e os ciclos internos, não os tokens da sequência. Na geração, cada novo token continua autoregressivo e usa um cache das convoluções causais. Esta mudança altera o vocabulário, o formato do embedding e as larguras das projeções; checkpoints anteriores não são compatíveis, então comece um treino novo sem `--resume`.

O Modelo Principal usa 48 experts densos por router: todos são calculados e combinados com pesos softmax que somam 1. Também usa conexões residuais experimentais inspiradas em mHC (mistura de quatro streams com matriz projetada por Sinkhorn para ser doubly stochastic) e Attention Residuals entre profundidades. Attencioner e De-vectorizer usam o mesmo módulo híbrido nos estágios internos. Esta é uma adaptação compacta para esta arquitetura, não uma reprodução integral dos artigos originais; os ganhos precisam ser medidos. A contagem de parâmetros aumentou e é impressa no começo do treino.

O Modelo Principal também inclui dois passos recursivos compartilhados inspirados no Tiny Recursive Model, refinando estados latentes de raciocínio e resposta, e oito ciclos internos inspirados no Continuous Thought Machine. Os pesos temporais por neurônio usam sempre uma janela fixa de oito posições: posições sem histórico são preenchidas com zero, e a normalização permanece sobre os oito pesos em todos os ticks. A sincronização estima covariância temporal para pares de neurônios amostrados globalmente, sem pressupor que índices vizinhos tenham relação especial. Use `--no-ctm-synchrony` para desativá-la e comparar por ablação; checkpoints da versão 13 registram essa opção, o mapa de tokens, o rank do embedding fatorado e as dimensões de cada componente. São adaptações para previsão autoregressiva de texto; os resultados publicados pelos trabalhos originais não garantem ganhos neste modelo.

## Componentes e tamanho

- **Embedding compartilhado fatorado:** fatores de `tamanho_do_tokenizer × 48` e 48 × 256, usados na entrada e na saída; aparece separadamente no relatório de parâmetros.
- **Attencioner:** projeção do embedding de 256 dimensões para a largura configurada, com fake quantization int4 nos vetores resultantes.
- **Modelo Principal:** convoluções causais paralelas, soft-router denso, mHC, Attention Residuals, CTM e refinamento recursivo inspirado no TRM.
- **De-vectorizer:** projeção fatorada do estado para o vocabulário.
- A contagem depende das constantes e do tokenizer. O programa imprime a contagem real e interrompe se ultrapassar o limite de 256 milhões.

A tabela anterior tinha 20.000 × 128 = 2.560.000 parâmetros. A fatoração atual, para o GPT-2 com UNK, tem 50.258 × 48 + 48 × 256 = 2.424.672, liberando 135.328 parâmetros. Quase metade foi alocada a cada projeção ao elevar `ATTENCIONER_WIDTH` e `DEVECTORIZER_WIDTH` de 192 para 231: cada uma ganha 67.119 parâmetros. A diferença restante em relação à divisão exata é 1.090 parâmetros no total.

Para configurar largura e profundidade separadamente, edite as constantes no início de `model.py`: `ATTENCIONER_WIDTH`, `ATTENCIONER_DEPTH`, `DEVECTORIZER_WIDTH` e `DEVECTORIZER_DEPTH`. A largura define as unidades internas de cada MLP; a profundidade define quantas camadas ocultas ele tem. Ambos continuam conectados à largura do Modelo Principal e ao embedding compartilhado de 256 dimensões. `EMBEDDING_RANK` define o rank da fatoração e está em 48. Mudanças nessas constantes alteram a contagem de parâmetros e exigem um treino novo.

O embedding de entrada é compartilhado com a matriz de saída do De-vectorizer. Os checkpoints gerados pela arquitetura anterior não são compatíveis e o carregador informa isso.

O alvo é número de parâmetros, não tamanho do arquivo. Pesos são mantidos em float32 durante o treino; a quantização de 4 bits aplica-se às representações vetoriais do Attencioner e é simulada com gradiente aproximado. Ela não quantiza os pesos do modelo.

## Requisitos

- Python 3.11 ou 3.12 recomendado.
- PyTorch 2.9 or newer (the script uses the built-in `torch.optim.Muon`).
- `transformers` para carregar/salvar o tokenizer público GPT-2.
- `datasets` para ler Cosmopedia em streaming.
- GPU CUDA recomendada. Memória e velocidade dependem da largura, profundidade, número de experts e tamanho do lote; o CPU pode ser muito lento.

Instalação:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers datasets
```

Para Windows, crie/ative o ambiente com `py -3.12 -m venv .venv` e `.venv\\Scripts\\Activate.ps1`. Para CUDA, use o comando PyTorch correspondente ao sistema e à versão CUDA disponível.

## Treinar em Cosmopedia

O padrão é o subconjunto `stories` de `HuggingFaceTB/cosmopedia`, lido em streaming, sem baixar antecipadamente o dataset inteiro:

```bash
python train.py --out vslm-256m.pt --steps 10000
```

Para comparar o CTM sem o cálculo e a projeção de sincronização, desative esse caminho:

```bash
python train.py --out vslm-sem-sincronia.pt --steps 10000 --no-ctm-synchrony
```

Ao retomar um checkpoint, mantenha a mesma opção de sincronização usada no início do treino.

Como o dataset é lido em streaming, `--epochs` usa um orçamento fixo de tokens, em vez de esperar o stream terminar. O padrão é 1 milhão de tokens de entrada por época; com `--batch-size 1 --block-size 128`, isso corresponde a 7.813 passos:

```bash
python train.py --out vslm-256m.pt --epochs 3
```

Altere o orçamento com `--tokens-per-epoch`. Ao retomar com `--resume`, o número informado em `--epochs` conta épocas adicionais, cada uma com esse orçamento.

O tokenizer público padrão é `gpt2`. Para selecionar outro subconjunto publicado do Cosmopedia:

```bash
python train.py --subset auto_math_text --out vslm-math.pt
```

## Otimizadores

O treino usa Muon nas projeções internas do Modelo Principal, Attencioner e De-vectorizer. Convoluções, embeddings, matriz de saída do vocabulário, vieses e normalizações usam AdamW. O clipping global de gradiente continua com limite 1,0. A variante MuonClip descrita para Transformers também faz QK-Clip em matrizes de query/key; esta arquitetura não possui atenção nem matrizes Q/K, então essa parte específica não se aplica.

O código usa a coluna `text`, acrescenta EOS entre documentos e treina causalmente: os logits em cada posição são comparados com o token seguinte. O dataset completo informado no Hub tem dezenas de milhões de amostras e cerca de 92 GB, por isso streaming evita um download completo, mas o tempo de treino depende dos passos, hardware e rede.

O checkpoint mais recente é atualizado a cada **100 passos por padrão**, ao receber Ctrl+C e ao final, no caminho `--out`. Um segundo arquivo `<nome>.best.pt` guarda o checkpoint com menor loss de validação. Por padrão, 128 documentos iniciais do stream são reservados para validação; até 16 lotes são avaliados no primeiro passo e a cada 500 passos. Configure isso com `--validation-documents`, `--validation-batches` e `--validation-every`. Ambos os checkpoints guardam pesos, passo e estados de Muon/AdamW; as gravações são atômicas.

Ao iniciar o treino, `train.py` abre o `code.html` no Firefox por meio de um servidor local. A página atualiza loss de treino a cada passo, loss de validação, melhor loss, contagem de melhorias, progresso e gráfico. Todo passo é acrescentado e descarregado no arquivo `<nome>.losses.jsonl`; ao retomar, use o mesmo `--out` ou informe o caminho antigo com `--loss-log` para continuar o mesmo histórico. Se o histórico estiver além do checkpoint retomado, ele é alinhado ao passo salvo para evitar pontos duplicados ou fora de ordem. O painel converte os dados históricos embutidos no `code.html` para JSONL ao retomar checkpoints antigos sem arquivo de histórico. O número de passos por época é calculado pela configuração atual (`--tokens-per-epoch`, `--batch-size` e `--block-size`) e exibido junto ao progresso da execução retomada.

Use `--best-out` para escolher outro caminho para o arquivo de menor loss.

Para continuar de um checkpoint:

```bash
python train.py --resume vslm-256m.pt --out vslm-256m.pt --steps 5000
```

`--steps` e `--epochs` nessa execução significam trabalho adicional. A arquitetura v13 não migra checkpoints anteriores porque mudou o tamanho do vocabulário, o formato do embedding e as larguras das projeções; comece um treino novo sem `--resume`. O mapa de tokens é salvo no checkpoint; o stream do Cosmopedia recomeça a leitura quando o processo é reiniciado.

Parâmetros de memória/treino:

```bash
python train.py --block-size 128 --batch-size 1 --steps 10000 --save-every 100
```

`block-size` e `batch-size` maiores aumentam o consumo de memória e o trabalho paralelo por lote. O caminho de treino não tem varredura recorrente por token; a geração continua sequencial porque cada previsão depende do token anterior.

`--compile` ativa `torch.compile` para o forward. É opcional: a compilação inicial pode demorar e o ganho varia conforme PyTorch, hardware e formato dos lotes.

## Gerar texto

```bash
python train.py generate --model vslm_corigido.pt --prompt "A ciência começa quando" --new-tokens 200
```

## Arquivos

- `model.py`: arquitetura, checkpoints e geração.
- `train.py`: treino, leitura streaming do Cosmopedia, otimizadores e validação.
- `requirements.txt`: dependências Python.
