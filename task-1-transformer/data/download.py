"""下载 ChnSentiCorp 中文情感分类数据集到当前目录。

依赖：pip install datasets pyarrow requests

优先尝试顺序：
  1. HF 镜像（HF_ENDPOINT 设置后）：从 hf-mirror.com 下载 Arrow 文件并转 Parquet
  2. HF 直连：同上流程
  3. ModelScope（备选）

下载完成后会在 data/ 下生成 train/validation/test.parquet 共 3 个文件。
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent


def download_via_arrow(base_url: str, label: str) -> bool:
    """从指定 base_url 下载 Arrow 文件，转成 parquet。"""
    import io
    import requests
    import pyarrow as pa
    import pandas as pd

    files = {
        'train':      'chn_senti_corp-train.arrow',
        'validation': 'chn_senti_corp-validation.arrow',
        'test':       'chn_senti_corp-test.arrow',
    }
    ok = True
    for split, fname in files.items():
        url = f'{base_url}/{fname}'
        print(f'  {label} {split}...', end=' ', flush=True)
        try:
            r = requests.get(url, timeout=60)
            if not r.ok:
                print(f'FAIL ({r.status_code})')
                ok = False
                continue
            with open(DATA_DIR / f'{split}.arrow', 'wb') as f:
                f.write(r.content)
            # 转 parquet
            with pa.OSFile(DATA_DIR / f'{split}.arrow', 'rb') as af:
                reader = pa.ipc.open_stream(af)
                table = reader.read_all()
            df = table.to_pandas()
            df.to_parquet(DATA_DIR / f'{split}.parquet')
            print(f'OK ({len(df)} rows)')
        except Exception as e:
            print(f'FAIL ({e})')
            ok = False
    return ok


def main():
    if 'HF_ENDPOINT' not in os.environ:
        print('[提示] 如下载缓慢，可先 export HF_ENDPOINT=https://hf-mirror.com 再重试\n')

    # 1. HF 镜像（国内推荐）
    mirror = os.environ.get('HF_ENDPOINT', '').rstrip('/')
    if mirror:
        mirror_base = f'{mirror}/datasets/seamew/ChnSentiCorp/resolve/main'
        print(f'尝试 HF 镜像：{mirror}')
        if download_via_arrow(mirror_base, '镜像'):
            print(f'\n[Done] 数据保存在 {DATA_DIR}')
            return

    # 2. HF 直连
    print('尝试 HF 直连...')
    if download_via_arrow('https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main', 'HF'):
        print(f'\n[Done] 数据保存在 {DATA_DIR}')
        return

    # 3. ModelScope 备选
    print('\n[失败] HF 直连与镜像均不可用。请尝试 ModelScope：')
    print('  pip install modelscope addict')
    print('  python -c "from modelscope.msdatasets import MsDataset; '
          "ds = MsDataset.load('damo/nlp_corpus', subset_name='ChnSentiCorp', split='train'); "
          'print(ds)"')
    sys.exit(1)


if __name__ == '__main__':
    main()