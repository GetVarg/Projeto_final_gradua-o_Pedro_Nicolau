# Laboratorio Mininet + FRRouting

Este laboratorio recria a topologia `gabriel/10/0` do dataset convertido:

`../dataset/topologias_convertidas/gabriel/10/0.json`

Ele cria:

- 10 roteadores Mininet (`r0` ... `r9`) rodando FRR `zebra`, `ospfd`, `staticd` e `bgpd`
- 10 hosts (`h_0_1` ... `h_9_1`)
- enlaces ponto-a-ponto `10.0.x.0/30`
- LANs `172.16.x.0/24`
- OSPF area `0.0.0.0` em todas as interfaces dos roteadores

## Build

Execute a partir de `Ic-llmToNetworkConfig/ibn_langgraph`:

```bash
docker build -t tcc-mininet-frr -f mininet_frr_lab/Dockerfile .
```

## Executar

Mininet precisa de privilegios de rede do kernel:

```bash
docker run --rm -it --privileged \
  --name tcc-mininet-frr \
  -v /lib/modules:/lib/modules:ro \
  tcc-mininet-frr
```

Voce entrara no prompt do Mininet. Exemplos uteis:

```text
mininet> nodes
mininet> net
mininet> h_0_1 ping -c 3 172.16.9.10
mininet> r0 ip route
mininet> r0 cat /tmp/tcc-mininet-frr/r0/ospfd.log
mininet> exit
```

## Usar outro JSON

Tambem da para montar outra topologia convertida sem rebuild:

```bash
docker run --rm -it --privileged \
  -v /lib/modules:/lib/modules:ro \
  -v "$PWD/dataset/topologias_convertidas/gabriel/10/0.json:/lab/topology.json:ro" \
  tcc-mininet-frr
```

Ou informe explicitamente o caminho dentro do container:

```bash
docker run --rm -it --privileged \
  -v /lib/modules:/lib/modules:ro \
  -v "$PWD/dataset:/dataset:ro" \
  tcc-mininet-frr \
  python3 /lab/run_topology.py --topology /dataset/topologias_convertidas/gabriel/10/0.json
```

## Aplicar uma saida do modelo

O script `apply_model_output.py` inicia a topologia, aplica os itens de
`test_outputs[].actual_commands` nos dispositivos correspondentes e salva um
relatorio com a saida e o codigo de retorno de cada comando.

No PowerShell, execute a partir de `Ic-llmToNetworkConfig/ibn_langgraph`:

```powershell
New-Item -ItemType Directory -Force mininet_frr_lab/results | Out-Null

docker run --rm --privileged `
  -v "${PWD}/outputs/meta-llama_Llama-3.1-8B-Instruct/pipeline_20260610_163248_correct.json:/lab/model_output.json:ro" `
  -v "${PWD}/mininet_frr_lab/results:/results" `
  tcc-mininet-frr `
  python3 /lab/apply_model_output.py
```

Para aplicar apenas um caso e evitar acumular configuracoes de outros casos:

```powershell
docker run --rm --privileged `
  -v "${PWD}/outputs/meta-llama_Llama-3.1-8B-Instruct/pipeline_20260610_163248_correct.json:/lab/model_output.json:ro" `
  -v "${PWD}/mininet_frr_lab/results:/results" `
  tcc-mininet-frr `
  python3 /lab/apply_model_output.py --case-id B02
```

## Observacoes

- Os nomes das interfaces seguem o JSON, por exemplo `0-eth0`, `4-eth0` e `h_0_1-eth0`.
- Os roteadores no Mininet usam nomes validos de host Linux: `r0`, `r1`, etc. O script mantem o ID original do dataset internamente.
- Os arquivos gerados do FRR ficam em `/tmp/tcc-mininet-frr` dentro do container.
