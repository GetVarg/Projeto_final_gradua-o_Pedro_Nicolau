# Cisco IOS Lab

Este laboratorio reaproveita a mesma topologia JSON usada pelo Mininet + FRR,
mas gera configuracoes iniciais Cisco IOS e aplica `actual_commands` em
roteadores Cisco/Cisco-like externos, como CML, GNS3 ou EVE-NG.

## Gerar configs IOS iniciais

Execute a partir de `Ic-llmToNetworkConfig/ibn_langgraph`:

```powershell
python .\cisco_lab\topology_to_cisco.py
```

Saidas geradas:

- `cisco_lab/generated/configs/R0.cfg`, `R1.cfg`, ...
- `cisco_lab/generated/topology_map.json`
- `cisco_lab/generated/inventory.example.json`

Importe ou cole os arquivos `.cfg` nos roteadores IOS do emulador.

## Interfaces

Por padrao, o conversor mapeia:

```text
0-eth0 -> GigabitEthernet0/0
0-eth1 -> GigabitEthernet0/1
0-eth2 -> GigabitEthernet0/2
```

Se sua imagem Cisco usa outro padrao, gere novamente:

```powershell
python .\cisco_lab\topology_to_cisco.py --interface-prefix "Ethernet0/"
```

## Inventario de gerenciamento

Copie `inventory.example.json` para `inventory.json` e preencha `host`, `port`,
`username`, `password` e `enable_password` conforme o console Telnet/SSH do seu
emulador.

Exemplo:

```json
{
  "defaults": {
    "protocol": "telnet",
    "username": "",
    "password": "",
    "enable_password": "",
    "command_timeout": 10
  },
  "devices": {
    "0": {
      "hostname": "R0",
      "host": "127.0.0.1",
      "port": 5000
    }
  }
}
```

## Aplicar comandos Cisco do modelo

O aplicador usa o mesmo contrato do lab FRR: ele le
`test_outputs[].actual_commands` do JSON de saida do modelo.

```powershell
python .\cisco_lab\apply_model_output_cisco.py `
  --model-output .\outputs\meta-llama_Llama-3.1-8B-Instruct\pipeline_20260625_122130_correct.json `
  --inventory .\cisco_lab\generated\inventory.json `
  --case-id B02
```

O relatorio padrao sai em:

```text
cisco_lab/results/model_command_report_cisco.json
```

## Observacoes

- Este lab nao sobe roteadores Cisco sozinho. Ele assume que CML, GNS3, EVE-NG
  ou outro emulador ja esta rodando os roteadores.
- Telnet nao exige dependencia extra. SSH exige o pacote opcional `paramiko`.
- O comando gerado no formato `configure terminal ; interface ... ; ...` e
  separado em linhas antes de ser enviado ao IOS.
