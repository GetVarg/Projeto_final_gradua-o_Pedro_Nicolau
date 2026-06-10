import json
import os
import re
import time
import unittest

from openai import OpenAI


MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct:novita")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
HF_ROUTER_BASE_URL = os.environ.get("HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1")

SYSTEM_PROMPT = """
ROLE
You analyze a high-level network intent and identify the distinct network-management objectives that the operator wants to achieve.

Your output is a list of atomic subintents, where each subintent represents one complete objective of the original intent.

A subintent is not a grammatical fragment.
A subintent is not an isolated parameter.
A subintent is not a topology entity.
A subintent is one desired network outcome that can later be validated, planned, and executed by the IBN pipeline.

REASONING PROCEDURE

Step 1: Identify network-effect verbs.
Find only actions that create, remove, modify, enable, disable, apply, bind, restart, or configure network state.

Step 2: Build one complete operation around each network-effect verb.
Attach all required complements to the operation, including device, interface, prefix, next-hop, gateway, IP, MAC, rate, protocol, direction, AS number, peer-group name, neighbor, or topology reference.

Step 3: Reject grammatical fragments.
Do not create a subintent from a phrase that only starts with or expresses:
"on ...", "to ...", "via ...", "using ...", "with ...", "as ...", "connected to ...", "mapping ...", or "pointing to ...".

Step 4: Split only independent network effects.
Create multiple subintents only when there are multiple independent effects.

Step 5: Preserve completeness.
Each subintent must be a complete network task. If a candidate subintent is incomplete alone, merge it with the operation it supports.

EXAMPLE OF THE DESIRED REASONING

Input intent:
"Configure a static route on router 3 to 172.16.4.0/24 via 10.0.4.2."

Step 1: Identify network-effect verbs.
- "Configure a static route" is the only network-effect action.

Step 2: Attach required complements.
- "on router 3" tells where the route is configured.
- "to 172.16.4.0/24" tells the destination prefix.
- "via 10.0.4.2" tells the next-hop.
These are required arguments of the static route operation.

Step 3: Reject grammatical fragments.
The following are not independent subintents:
- "on router 3"
- "to 172.16.4.0/24"
- "via 10.0.4.2"

Step 4: Decide split.
There is only one independent network effect: configuring a static route.

Correct final output:
{
  "subintents": [
    {
      "id": "S1",
      "text": "Configure a static route on router 3 to 172.16.4.0/24 via 10.0.4.2"
    }
  ]
}

Incorrect output:
{
  "subintents": [
    {
      "id": "S1",
      "text": "Configure a static route"
    },
    {
      "id": "S2",
      "text": "on router 3"
    },
    {
      "id": "S3",
      "text": "to 172.16.4.0/24"
    },
    {
      "id": "S4",
      "text": "via 10.0.4.2"
    }
  ]
}

Why incorrect:
This splits arguments of the same route operation into grammatical fragments. Only the first item contains a network effect; the others are incomplete parameters.

OUTPUT CONTRACT

Return exactly one JSON object:
{
  "subintents": [
    {
      "id": "S1",
      "text": "..."
    }
  ]
}

OUTPUT RULES
- Return JSON only.
- Do not include reasoning in the final output.
- Use sequential IDs: S1, S2, S3, ...
- Preserve explicit values from the original intent.
- If uncertain, prefer fewer subintents.
""".strip()

INTENTS = [
    "Configure a static route on router 3 to 172.16.4.0/24 via 10.0.4.2.",
    "Enable IPv4 forwarding globally on router 7.",
    "Administratively shut down interface 8-eth0 on router 8.",
    "Configure a static ARP entry on router 0 mapping IP 10.0.0.2 to MAC address 02:00:00:00:00:02.",
    "Configure a static route on router 0 to 172.16.9.0/24 using the IP address of the neighbor connected to interface 0-eth2 as next-hop.",
    "Configure a static route on router 2 to 172.16.7.0/24 using the IP of the device connected to 2-eth1 as next-hop.",
    "Configure a static route on router 1 to reach the LAN subnet of router 5, using the IP of the neighbor connected to 1-eth1 as the gateway.",
    "Apply a 50Mbps rate limit for inbound UDP traffic on the specific interface of router 4 that connects to router 0.",
    "Find the IPv4 network CIDR assigned to interface 3-eth1 and configure a static route on router 0 pointing to that network via 10.0.1.2.",
    "Create a BGP peer-group named IBGP on router 0 and bind neighbor 10.0.1.2 to it.",
    "Configure a BGP peer-group named IBGP on router 0, set its remote-as to 65000, and bind neighbor 10.0.1.2 to it.",
    "Set MED to 100 for BGP neighbor 10.0.2.2 and enable AS-path multipath on router 0.",
    "Enable Jumbo frames by setting the MTU to 9216 on interface 8-eth0 of router 8, and restart the interface by bringing it down then up.",
    "Find the router connected to interface 0-eth1, and configure a static route on that remote router back to 172.16.0.0/24 via 10.0.1.1.",
    "Disable ARP, shut down interface 7-eth0, and remove its IPv4 address on router 7.",
]


def _first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("response does not contain a JSON object")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:].strip()


def call_hf_router(intent: str) -> tuple[dict, str, float]:
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN before running this test.")

    client = OpenAI(base_url=HF_ROUTER_BASE_URL, api_key=HF_TOKEN)
    user_payload = json.dumps({"intent": intent}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_payload}\n\n"
                "Return only the JSON object. Start the answer with `{` and end it with `}`. "
                "Do not include reasoning, markdown, prose, or any text after the JSON."
            ),
        },
    ]

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )
    elapsed = time.perf_counter() - start
    raw = (response.choices[0].message.content or "").strip()
    parsed = json.loads(_first_balanced_json_object(raw))
    return parsed, raw, elapsed


def validate_result(parsed: dict) -> list[str]:
    errors = []
    subintents = parsed.get("subintents")
    if not isinstance(subintents, list) or not subintents:
        return ["subintents must be a non-empty list"]

    for index, subintent in enumerate(subintents, start=1):
        if not isinstance(subintent, dict):
            errors.append(f"S{index} is not an object")
            continue
        expected_id = f"S{index}"
        if subintent.get("id") != expected_id:
            errors.append(f"expected id {expected_id}, got {subintent.get('id')!r}")
        text = subintent.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{expected_id}.text must be a non-empty string")

    dependent_only = re.compile(
        r"^(find|identify|determine|specify|using|via|as next-hop|as the gateway|connected to)\b",
        re.IGNORECASE,
    )
    for subintent in subintents:
        text = str(subintent.get("text") or "").strip()
        if dependent_only.search(text):
            errors.append(f"dependent clause became standalone subintent: {text!r}")

    return errors


class DiscretizeEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = []
        for index, intent in enumerate(INTENTS, start=1):
            start = time.perf_counter()
            raw = ""
            try:
                parsed, raw, elapsed = call_hf_router(intent)
                parse_errors = validate_result(parsed)
                subtexts = [item.get("text") for item in parsed.get("subintents", []) if isinstance(item, dict)]
            except Exception as exc:
                elapsed = time.perf_counter() - start
                parse_errors = [f"{type(exc).__name__}: {exc}"]
                subtexts = []
            result = {
                "id": index,
                "intent": intent,
                "elapsed_sec": round(elapsed, 3),
                "subintent_count": len(subtexts),
                "subintents": subtexts,
                "raw": raw,
                "errors": parse_errors,
            }
            cls.results.append(result)
            print("\n" + "=" * 100)
            print(f"[DISCRETIZE ENDPOINT][INTENT {index}] {intent}")
            print("=" * 100)
            print(json.dumps(result, ensure_ascii=False, indent=2))

    def test_all_outputs_are_valid_subintent_json(self):
        failures = {
            result["id"]: result["errors"]
            for result in self.results
            if result["errors"]
        }
        self.assertEqual({}, failures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
