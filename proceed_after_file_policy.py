%%bash
python - <<'PY'
import pathlib
p = pathlib.Path('vps/vpscore/config.py')
s = p.read_text()
s = s.replace('Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-3B-Instruct')
p.write_text(s)
print("Model -> Qwen/Qwen2.5-3B-Instruct ")
PY




%%bash
python - <<'PY'
import pathlib, re
p = pathlib.Path('vps/vpscore/config.py')
s = p.read_text()
s = re.sub(r'dtype:\s*Literal\["bf16","fp16","fp32"\]\s*=\s*"bf16"',
           'dtype: Literal["bf16","fp16","fp32"] = "fp16"', s)
p.write_text(s)
print("VPSConfig.dtype -> fp16 ")
PY

