# ============================================
# ГИПОТЕЗА 5: Влияние многодетности на здоровье детей
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ГИПОТЕЗА 5: Влияние многодетности на здоровье детей")
print("=" * 80)

# ============================================
# ШАГ 1: Загрузка данных
# ============================================

print("\n[1] Загрузка данных...")
df = pd.read_csv('Final.csv')
print(f"Загружено записей: {len(df)}")
print(f"Колонок в датасете: {df.shape[1]}")

# ============================================
# ШАГ 2: Создание переменных для анализа
# ============================================

print("\n[2] Создание переменных для анализа...")

# Анемия у ребенка (Hg_child_adjust в г/л, порог 110 г/л)
df['child_anemia'] = (df['Hg_child_adjust'] < 110).astype(int)
df['child_anemia_level'] = pd.cut(df['Hg_child_adjust'], 
                                  bins=[0, 70, 100, 110, 200],
                                  labels=['Тяжелая (<70)', 'Умеренная (70-100)', 
                                          'Легкая (100-110)', 'Нет анемии (>110)'],
                                  right=False)
print(f"[+] Анемия у ребенка: {df['child_anemia'].mean()*100:.1f}%")
print(f"    Распределение по уровням:")
print(df['child_anemia_level'].value_counts().sort_index().to_string())

# Полная вакцинация
df['full_vaccination'] = ((df['DPT_full'] == 1) & (df['MEASLES_full'] == 1)).astype(int)
print(f"[+] Полная вакцинация: {df['full_vaccination'].mean()*100:.1f}%")

# Вес при рождении (очистка)
df['birth_weight_clean'] = df['Birth_Weight'].apply(
    lambda x: x if 1.5 <= x <= 5.0 else np.nan
)
print(f"[+] Вес при рождении: {df['birth_weight_clean'].notna().sum()} записей")
print(f"    Средний вес: {df['birth_weight_clean'].mean():.3f} кг")

# Категории многодетности
print(f"\n[+] Tot_child_born статистика:")
print(f"    Среднее: {df['Tot_child_born'].mean():.2f}")
print(f"    Медиана: {df['Tot_child_born'].median():.2f}")
print(f"    Макс: {df['Tot_child_born'].max()}")

df['family_size_cat'] = pd.cut(df['Tot_child_born'], 
                                bins=[0, 2, 4, 50], 
                                labels=['Малодетные (1-2)', 'Среднедетные (3-4)', 'Многодетные (5+)'],
                                right=True)

print("\n    Распределение по категориям:")
print(df['family_size_cat'].value_counts().sort_index().to_string())

# Индикатор детской смертности
df['child_mortality'] = ((df['Sons_died'] > 0) | (df['Daughters_died'] > 0)).astype(int)
print(f"\n[+] Индикатор детской смертности:")
print(f"    Семьи с опытом смерти детей: {df['child_mortality'].sum()} ({df['child_mortality'].mean()*100:.1f}%)")

# Интегральный показатель "репродуктивного стресса"
df['reproductive_stress'] = (
    pd.cut(df['Tot_child_born'], bins=[0, 2, 4, 50], labels=[0, 1, 2]).astype(int) + 
    df['child_mortality']
)
print(f"    Интегральный показатель репродуктивного стресса (0-3):")
print(df['reproductive_stress'].value_counts().sort_index().to_string())

# Прокси интервалов между рождениями
df['child_per_year'] = df['Tot_child_born'] / (df['Res_Age'] + 1)
df['high_birth_rate'] = (df['child_per_year'] > df['child_per_year'].median()).astype(int)
print(f"\n[+] Прокси интервалов между рождениями:")
print(f"    Высокая интенсивность рождений: {df['high_birth_rate'].mean()*100:.1f}%")

# ============================================
# ШАГ 3: Статистика по группам многодетности
# ============================================

print("\n" + "=" * 80)
print("[3] Статистика по группам многодетности")
print("=" * 80)

print("\nДемографические характеристики по группам:")

for var, label in [('Res_Age', 'Возраст матери'), 
                   ('Married_age', 'Возраст первого брака'),
                   ('Edu_level', 'Образование'),
                   ('Wealth_Idx_Lb', 'Благосостояние')]:
    stats = df.groupby('family_size_cat')[var].agg(['mean', 'std']).round(2)
    print(f"\n{label}:")
    print(stats.to_string())

print("\n" + "-" * 40)
print("ПОКАЗАТЕЛИ ЗДОРОВЬЯ ДЕТЕЙ ПО ГРУППАМ:")
print("-" * 40)

# Вес при рождении
weight_stats = df.groupby('family_size_cat')['birth_weight_clean'].agg(['mean', 'sem', 'count'])
print("\nВес при рождении (кг):")
for cat in weight_stats.index:
    mean = weight_stats.loc[cat, 'mean']
    ci = 1.96 * weight_stats.loc[cat, 'sem']
    print(f"  {cat}: {mean:.3f} (95% ДИ: {mean-ci:.3f}-{mean+ci:.3f})")

# Анемия
anemia_stats = df.groupby('family_size_cat')['child_anemia'].agg(['mean', 'count'])
print("\nАнемия (%):")
for cat in anemia_stats.index:
    p = anemia_stats.loc[cat, 'mean'] * 100
    n = anemia_stats.loc[cat, 'count']
    ci = 1.96 * np.sqrt((p/100 * (1-p/100)) / n) * 100
    print(f"  {cat}: {p:.1f}% (95% ДИ: {p-ci:.1f}%-{p+ci:.1f}%)")

# Вакцинация
vacc_stats = df.groupby('family_size_cat')['full_vaccination'].agg(['mean', 'count'])
print("\nВакцинация (%):")
for cat in vacc_stats.index:
    p = vacc_stats.loc[cat, 'mean'] * 100
    n = vacc_stats.loc[cat, 'count']
    ci = 1.96 * np.sqrt((p/100 * (1-p/100)) / n) * 100
    print(f"  {cat}: {p:.1f}% (95% ДИ: {p-ci:.1f}%-{p+ci:.1f}%)")

# Детская смертность
mortality_stats = df.groupby('family_size_cat')['child_mortality'].agg(['mean', 'count'])
print("\nДетская смертность (%):")
for cat in mortality_stats.index:
    p = mortality_stats.loc[cat, 'mean'] * 100
    n = mortality_stats.loc[cat, 'count']
    ci = 1.96 * np.sqrt((p/100 * (1-p/100)) / n) * 100
    print(f"  {cat}: {p:.1f}% (95% ДИ: {p-ci:.1f}%-{p+ci:.1f}%)")

# ============================================
# ШАГ 4: Сравнение групп (ANOVA и хи-квадрат)
# ============================================

print("\n" + "=" * 80)
print("[4] СРАВНЕНИЕ ГРУПП (ANOVA и ХИ-КВАДРАТ)")
print("=" * 80)

# --- Анализ 1: Вес при рождении (ANOVA) ---
print("\n--- СРАВНЕНИЕ ВЕСА ПРИ РОЖДЕНИИ ---")

groups = []
group_names = []
for cat in df['family_size_cat'].cat.categories:
    group_data = df[df['family_size_cat'] == cat]['birth_weight_clean'].dropna()
    groups.append(group_data)
    group_names.append(cat)
    print(f"{cat}: n={len(group_data)}, средний вес={group_data.mean():.3f} кг")

f_stat, p_val = f_oneway(*groups)
print(f"\nРезультаты ANOVA:")
print(f"  F-статистика = {f_stat:.3f}")
print(f"  p-value = {p_val:.4f}")

if p_val < 0.05:
    print("  [*] Статистически значимые различия обнаружены")
    
    print("\n  Post-hoc анализ (тест Тьюки):")
    tukey_data = df[['birth_weight_clean', 'family_size_cat']].dropna()
    tukey_results = pairwise_tukeyhsd(tukey_data['birth_weight_clean'], 
                                      tukey_data['family_size_cat'])
    
    # Форматируем вывод
    results_data = []
    for i in range(len(tukey_results._results_table.data) - 1):
        row = tukey_results._results_table.data[i + 1]
        results_data.append([row[0], row[1], row[2], row[3]])
    
    results_df = pd.DataFrame(results_data, 
                             columns=['Группа 1', 'Группа 2', 'Разница средних', 'p-value'])
    print(results_df.to_string(index=False))
else:
    print("  [-] Статистически значимых различий не обнаружено")

# --- Анализ 2: Анемия (хи-квадрат) ---
print("\n--- СВЯЗЬ МНОГОДЕТНОСТИ И АНЕМИИ ---")

ct_anemia = pd.crosstab(df['family_size_cat'], df['child_anemia'])
ct_anemia.columns = ['Нет анемии', 'Есть анемия']

print("\nТаблица сопряженности:")
print(ct_anemia)

pct_anemia = pd.crosstab(df['family_size_cat'], df['child_anemia'], normalize='index') * 100
pct_anemia.columns = ['Нет анемии, %', 'Есть анемия, %']

print("\nДоля детей с анемией по группам (%):")
print(pct_anemia.round(1))

chi2, p_val, dof, expected = chi2_contingency(ct_anemia)

# Расчет V Крамера
n = ct_anemia.sum().sum()
v_cramer = np.sqrt(chi2 / (n * min(ct_anemia.shape[0]-1, ct_anemia.shape[1]-1)))

print(f"\nРезультаты теста хи-квадрат:")
print(f"  χ² = {chi2:.2f}")
print(f"  p-value = {p_val:.4f}")
print(f"  V Крамера = {v_cramer:.3f}")

# ============================================
# ШАГ 5: Корреляционный анализ
# ============================================

print("\n" + "=" * 80)
print("[5] КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 80)

corr_vars = ['Tot_child_born', 'birth_weight_clean', 'child_anemia', 
             'full_vaccination', 'Res_Age', 'Married_age', 'Edu_level', 
             'Wealth_Idx_Lb', 'child_mortality']

corr_df = df[corr_vars].dropna()
corr_matrix = corr_df.corr(method='pearson')

print("\nМатрица корреляций Пирсона:")
print(corr_matrix.round(3))

# Визуализация
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            annot_kws={'size': 10})
plt.title('Корреляционная матрица: многодетность и здоровье детей', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

print("\nКорреляции с количеством детей (Tot_child_born):")
correlations = corr_matrix['Tot_child_born'].drop('Tot_child_born').sort_values(ascending=False)
for var, corr in correlations.items():
    strength = "сильная" if abs(corr) > 0.5 else "средняя" if abs(corr) > 0.3 else "слабая"
    direction = "положительная" if corr > 0 else "отрицательная"
    print(f"  {var}: {corr:.3f} ({strength} {direction} связь)")

# Диаграмма рассеяния
plt.figure(figsize=(10, 6))
scatter_data = df[['Tot_child_born', 'birth_weight_clean']].dropna()
sns.regplot(data=scatter_data, x='Tot_child_born', y='birth_weight_clean',
           scatter_kws={'alpha':0.3, 'color':'blue'},
           line_kws={'color':'red', 'linewidth':2})
plt.xlabel('Общее число рожденных детей', fontsize=12)
plt.ylabel('Вес при рождении (кг)', fontsize=12)
plt.title('Связь между числом детей в семье и весом при рождении', fontsize=14)

from scipy.stats import pearsonr
corr, p_val = pearsonr(scatter_data['Tot_child_born'], scatter_data['birth_weight_clean'])
plt.text(0.05, 0.95, f'Корреляция: r={corr:.3f}\np-value: {p_val:.4f}',
        transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# ШАГ 6: Логистическая регрессия
# ============================================

print("\n" + "=" * 80)
print("[6] ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print("=" * 80)

target_var = 'child_anemia'
target_name = 'анемия у ребенка'

print(f"\nЦелевая переменная: {target_name}")

feature_cols = ['Tot_child_born', 'Res_Age', 'Married_age', 'Edu_level', 
                'Wealth_Idx_Lb', 'ResidenceType_Urban', 'child_mortality', 
                'high_birth_rate']

print(f"Признаки для регрессии: {feature_cols}")

model_df = df[[target_var] + feature_cols].dropna()
print(f"Размер выборки: {len(model_df)}")

X = model_df[feature_cols]
y = model_df[target_var]

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Модель 1: Только количество детей
X1 = X_scaled[['Tot_child_born']].values
model1 = LogisticRegression(max_iter=1000)
model1.fit(X1, y)
y_pred1 = model1.predict_proba(X1)[:, 1]
auc1 = roc_auc_score(y, y_pred1)

# Модель 2: Все факторы
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_scaled, y)
y_pred2 = model2.predict_proba(X_scaled)[:, 1]
auc2 = roc_auc_score(y, y_pred2)

print("\n" + "=" * 40)
print("РЕЗУЛЬТАТЫ РЕГРЕССИИ:")
print("=" * 40)

print(f"\nМодель 1 (только Tot_child_born):")
print(f"  Коэффициент: {model1.coef_[0][0]:.4f}")
or_value = np.exp(model1.coef_[0][0])
print(f"  Отношение шансов: {or_value:.3f}")
print(f"  AUC-ROC = {auc1:.4f}")

print(f"\nМодель 2 (все факторы):")
print(f"  AUC-ROC = {auc2:.4f}")
print(f"  Улучшение: +{(auc2-auc1)*100:.1f}%")

print("\nКоэффициенты и отношения шансов (модель 2):")
print("-" * 50)
print(f"{'Признак':<25} {'Коэф.':<10} {'Отнош. шансов':<15}")
print("-" * 50)

for name, coef in zip(feature_cols, model2.coef_[0]):
    or_val = np.exp(coef)
    print(f"{name:<25} {coef:>9.4f} {or_val:>14.3f}")

# ROC-кривые
fpr1, tpr1, _ = roc_curve(y, y_pred1)
fpr2, tpr2, _ = roc_curve(y, y_pred2)

plt.figure(figsize=(10, 8))
plt.plot(fpr1, tpr1, 'b-', linewidth=2, label=f'Модель 1 (только число детей, AUC={auc1:.3f})')
plt.plot(fpr2, tpr2, 'r-', linewidth=2, label=f'Модель 2 (все факторы, AUC={auc2:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Случайная модель (AUC=0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC-кривые для предсказания {target_name}', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# ШАГ 7: Стратифицированный анализ
# ============================================

print("\n" + "=" * 80)
print("[7] СТРАТИФИЦИРОВАННЫЙ АНАЛИЗ")
print("=" * 80)

# Стратификация по благосостоянию
wealth_median = df['Wealth_Idx_Lb'].median()
df['wealth_group'] = df['Wealth_Idx_Lb'].apply(
    lambda x: 'Низкое' if x <= wealth_median else 'Высокое'
)

# Стратификация по образованию
df['edu_group'] = df['Edu_level'].apply(
    lambda x: 'Низкое' if x <= 1 else 'Высокое'
)

print("\n" + "=" * 40)
print("РЕЗУЛЬТАТЫ ПО БЛАГОСОСТОЯНИЮ:")
print("=" * 40)

for var in ['child_anemia', 'full_vaccination', 'birth_weight_clean']:
    print(f"\n{var}:")
    result = df.groupby(['wealth_group', 'family_size_cat'])[var].agg(['mean', 'count']).round(3)
    if var in ['child_anemia', 'full_vaccination']:
        result['mean'] = result['mean'] * 100
    print(result.to_string())

print("\n" + "=" * 40)
print("РЕЗУЛЬТАТЫ ПО ОБРАЗОВАНИЮ:")
print("=" * 40)

for var in ['child_anemia', 'full_vaccination', 'birth_weight_clean']:
    print(f"\n{var}:")
    result = df.groupby(['edu_group', 'family_size_cat'])[var].agg(['mean', 'count']).round(3)
    if var in ['child_anemia', 'full_vaccination']:
        result['mean'] = result['mean'] * 100
    print(result.to_string())

# Визуализация стратификации
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Стратифицированный анализ: влияние многодетности в разных группах', fontsize=16, y=1.02)

health_vars = ['birth_weight_clean', 'child_anemia', 'full_vaccination']
var_titles = ['Вес при рождении (кг)', 'Анемия (%)', 'Вакцинация (%)']

# Графики по благосостоянию (верхний ряд)
for i, (var, title) in enumerate(zip(health_vars, var_titles)):
    plot_data = df.groupby(['wealth_group', 'family_size_cat'])[var].mean()
    if var in ['child_anemia', 'full_vaccination']:
        plot_data = plot_data * 100
    
    plot_df = plot_data.unstack()
    plot_df.plot(kind='bar', ax=axes[0, i], color=['#3498db', '#e74c3c', '#2ecc71'], 
                edgecolor='black', alpha=0.8)
    axes[0, i].set_xlabel('Благосостояние', fontsize=11)
    axes[0, i].set_ylabel(title, fontsize=11)
    axes[0, i].set_title(f'{title} по благосостоянию', fontsize=12)
    axes[0, i].legend(title='Размер семьи', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, i].grid(True, alpha=0.3, axis='y')
    axes[0, i].axhline(y=plot_data.mean(), color='red', linestyle='--', alpha=0.5)

# Графики по образованию (нижний ряд)
for i, (var, title) in enumerate(zip(health_vars, var_titles)):
    plot_data = df.groupby(['edu_group', 'family_size_cat'])[var].mean()
    if var in ['child_anemia', 'full_vaccination']:
        plot_data = plot_data * 100
    
    plot_df = plot_data.unstack()
    plot_df.plot(kind='bar', ax=axes[1, i], color=['#3498db', '#e74c3c', '#2ecc71'], 
                edgecolor='black', alpha=0.8)
    axes[1, i].set_xlabel('Образование', fontsize=11)
    axes[1, i].set_ylabel(title, fontsize=11)
    axes[1, i].set_title(f'{title} по образованию', fontsize=12)
    axes[1, i].legend(title='Размер семьи', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, i].grid(True, alpha=0.3, axis='y')
    axes[1, i].axhline(y=plot_data.mean(), color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ============================================
# ШАГ 8: Основные визуализации
# ============================================

print("\n[8] Визуализация основных результатов...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Влияние многодетности на здоровье детей', fontsize=16, y=1.02)

# График 1: Распределение семей
ax1 = axes[0, 0]
size_dist = df['family_size_cat'].value_counts().sort_index()
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(range(len(size_dist)), size_dist.values, color=colors, edgecolor='black', alpha=0.8)
ax1.set_xticks(range(len(size_dist)))
ax1.set_xticklabels(size_dist.index, rotation=45, ha='right')
ax1.set_ylabel('Количество семей')
ax1.set_title('Распределение семей по размеру')

for bar, val in zip(bars, size_dist.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(size_dist.values), 
            f'{val}', ha='center', va='bottom', fontsize=10)

# График 2: Вес при рождении
ax2 = axes[0, 1]
weight_by_size = df.groupby('family_size_cat')['birth_weight_clean'].agg(['mean', 'sem'])
x_pos = np.arange(len(weight_by_size))
bars = ax2.bar(x_pos, weight_by_size['mean'], yerr=1.96*weight_by_size['sem'],
               color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', 
               alpha=0.8, capsize=5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(weight_by_size.index, rotation=45, ha='right')
ax2.set_ylabel('Средний вес при рождении (кг)')
ax2.set_title('Вес при рождении по группам')
ax2.axhline(y=df['birth_weight_clean'].mean(), color='red', linestyle='--', 
            label=f'Среднее: {df["birth_weight_clean"].mean():.2f} кг')
ax2.legend()

for bar, (idx, row) in zip(bars, weight_by_size.iterrows()):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
            f'{row["mean"]:.2f}', ha='center', va='bottom', fontsize=10)

# График 3: Анемия
ax3 = axes[1, 0]
anemia_by_size = df.groupby('family_size_cat')['child_anemia'].agg(['mean', 'sem'])
anemia_by_size['mean'] = anemia_by_size['mean'] * 100
anemia_by_size['sem'] = anemia_by_size['sem'] * 100

x_pos = np.arange(len(anemia_by_size))
colors = ['#e74c3c' if x > anemia_by_size['mean'].mean() else '#3498db' for x in anemia_by_size['mean']]
bars = ax3.bar(x_pos, anemia_by_size['mean'], yerr=1.96*anemia_by_size['sem'],
               color=colors, edgecolor='black', alpha=0.8, capsize=5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(anemia_by_size.index, rotation=45, ha='right')
ax3.set_ylabel('Доля с анемией (%)')
ax3.set_title('Анемия у детей по группам')
ax3.axhline(y=anemia_by_size['mean'].mean(), color='red', linestyle='--', 
            label=f'Среднее: {anemia_by_size["mean"].mean():.1f}%')
ax3.legend()

for bar, (idx, row) in zip(bars, anemia_by_size.iterrows()):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 1, 
            f'{row["mean"]:.1f}%', ha='center', va='bottom', fontsize=10)

# График 4: Вакцинация
ax4 = axes[1, 1]
vacc_by_size = df.groupby('family_size_cat')['full_vaccination'].agg(['mean', 'sem'])
vacc_by_size['mean'] = vacc_by_size['mean'] * 100
vacc_by_size['sem'] = vacc_by_size['sem'] * 100

x_pos = np.arange(len(vacc_by_size))
colors = ['#2ecc71' if x > vacc_by_size['mean'].mean() else '#e74c3c' for x in vacc_by_size['mean']]
bars = ax4.bar(x_pos, vacc_by_size['mean'], yerr=1.96*vacc_by_size['sem'],
               color=colors, edgecolor='black', alpha=0.8, capsize=5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(vacc_by_size.index, rotation=45, ha='right')
ax4.set_ylabel('Доля вакцинированных (%)')
ax4.set_title('Вакцинация по группам')
ax4.axhline(y=vacc_by_size['mean'].mean(), color='red', linestyle='--', 
            label=f'Среднее: {vacc_by_size["mean"].mean():.1f}%')
ax4.legend()

for bar, (idx, row) in zip(bars, vacc_by_size.iterrows()):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 1, 
            f'{row["mean"]:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# ============================================
# ШАГ 9: Сводная таблица результатов
# ============================================

print("\n" + "=" * 80)
print("[9] СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)

summary_data = []

for cat in df['family_size_cat'].cat.categories:
    cat_data = df[df['family_size_cat'] == cat]
    
    row = {
        'Категория семьи': cat,
        'Количество': len(cat_data),
        'Среднее число детей': f"{cat_data['Tot_child_born'].mean():.2f}"
    }
    
    # Вес при рождении
    bw_mean = cat_data['birth_weight_clean'].mean()
    bw_ci = 1.96 * cat_data['birth_weight_clean'].sem()
    row['Вес при рождении (кг)'] = f"{bw_mean:.2f} (±{bw_ci:.2f})"
    
    # Анемия
    an_mean = cat_data['child_anemia'].mean() * 100
    an_ci = 1.96 * np.sqrt((an_mean/100 * (1-an_mean/100)) / len(cat_data)) * 100
    row['Анемия (%)'] = f"{an_mean:.1f} (±{an_ci:.1f})"
    
    # Вакцинация
    vacc_mean = cat_data['full_vaccination'].mean() * 100
    vacc_ci = 1.96 * np.sqrt((vacc_mean/100 * (1-vacc_mean/100)) / len(cat_data)) * 100
    row['Вакцинация (%)'] = f"{vacc_mean:.1f} (±{vacc_ci:.1f})"
    
    # Детская смертность
    mort_mean = cat_data['child_mortality'].mean() * 100
    mort_ci = 1.96 * np.sqrt((mort_mean/100 * (1-mort_mean/100)) / len(cat_data)) * 100
    row['Детская смертность (%)'] = f"{mort_mean:.1f} (±{mort_ci:.1f})"
    
    # Возраст матери
    row['Ср. возраст матери'] = f"{cat_data['Res_Age'].mean():.1f}"
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\nСводная таблица результатов (среднее ± 95% ДИ):")
print("=" * 100)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("АНАЛИЗ ПО ГИПОТЕЗЕ 5 ЗАВЕРШЕН")
print("=" * 80)