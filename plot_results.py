import json
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    with open('build/results.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Ошибка: Файл 'build/results.json' не найден.")
    print("Сначала запустите: ./benchmark_kernels --benchmark_format=json > results.json")
    exit(1)

matmul_data = defaultdict(lambda: ([], []))
conv_data = defaultdict(lambda: ([], []))

for bench in data['benchmarks']:
    name_parts = bench['name'].split('/')
    func_name = name_parts[0]
    size = int(name_parts[1])
    
    time_ms = bench['cpu_time'] / 1_000_000.0

    if "MatMul" in func_name:
        matmul_data[func_name][0].append(size)
        matmul_data[func_name][1].append(time_ms)
    elif "Conv" in func_name:
        conv_data[func_name][0].append(size)
        conv_data[func_name][1].append(time_ms)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels = {
    'BM_NaiveMatMul': ('Naive', 'o'),
    'BM_CacheFriendlyMatMul': ('Cache-Friendly', 's'),
    'BM_TiledMatMul': ('Tiled (Block=32)', '^'),
    'BM_AvxTiledMatMul': ('AVX + Tiled', 'd'),
    'BM_NaiveConv': ('Naive Conv', 'o'),
    'BM_Im2ColConvAdvanced': ('Im2Col + AVX', 'd')
}

for func, (sizes, times) in matmul_data.items():
    label, marker = labels.get(func, (func, 'o'))
    lw = 2.5 if 'Avx' in func else 1.5 
    ax1.plot(sizes, times, marker=marker, label=label, linewidth=lw)

ax1.set_title('Производительность матричного умножения', fontsize=14)
ax1.set_xlabel('Размер матрицы (N x N)', fontsize=12)
ax1.set_ylabel('Время выполнения (мс)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

for func, (sizes, times) in conv_data.items():
    label, marker = labels.get(func, (func, 'o'))
    lw = 2.5 if 'Im2Col' in func else 1.5
    ax2.plot(sizes, times, marker=marker, label=label, linewidth=lw)

ax2.set_title('Производительность свертки', fontsize=14)
ax2.set_xlabel('Размер изображения (H x W)', fontsize=12)
ax2.set_ylabel('Время выполнения (мс)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=300)
print("Графики успешно сгенерированы из JSON и сохранены в 'benchmark_results.png'")