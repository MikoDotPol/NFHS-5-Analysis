# ============================================
# ГИПОТЕЗА 1: Влияние образования матери на здоровье детей
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ГИПОТЕЗА 1: Влияние образования матери на здоровье детей")
print("=" * 80)

# ============================================
# ШАГ 1: Загрузка данных
# ============================================

print("\n[1] Загрузка данных...")
df = pd.read_csv('Final.csv')
print(f"Загружено записей: {len(df)}")

# ============================================
# ШАГ 2: Подготовка переменных
# ============================================

print("\n[2] Подготовка переменных...")

# АНЕМИЯ ДЕТЕЙ (Hg_child_adjust в г/л, порог < 110)
df['child_anemia'] = (df['Hg_child_adjust'] < 110).astype(int)
print(f"  Анемия у детей: {df['child_anemia'].mean()*100:.1f}%")

# Образование матери
edu_map = {0: 'Нет образования', 1: 'Начальное', 2: 'Среднее', 3: 'Высшее'}
df['edu_cat'] = df['Edu_level'].map(edu_map)
print("\n  Распределение образования матери:")
print(df['edu_cat'].value_counts().sort_index())

# Вакцинация детей от кори
df['measles_vacc'] = df['MEASLES_full']
print(f"\n  Вакцинация детей от кори: {df['measles_vacc'].mean()*100:.1f}%")

# Вес детей при рождении (очистка от выбросов)
df['birth_weight_clean'] = df['Birth_Weight'].apply(
    lambda x: x if 1.5 <= x <= 5.0 else np.nan
)
print(f"  Вес детей при рождении: средний {df['birth_weight_clean'].mean():.2f} кг")

# Доступ к информации (ТВ как прокси)
df['tv_access'] = df['House_tv']
print(f"  Доступ к ТВ: {df['tv_access'].mean()*100:.1f}% семей")

# Тип проживания
df['urban_flag'] = df['ResidenceType_Urban']
print(f"  Городское население: {df['urban_flag'].mean()*100:.1f}%")

# ============================================
# ШАГ 3: Подготовка аналитической выборки
# ============================================

print("\n[3] Подготовка выборки для анализа...")

# Включаем Hg_child_adjust в анализ
analysis_df = df[['edu_cat', 'Edu_level', 'child_anemia', 'Hg_child_adjust',
                  'measles_vacc', 'birth_weight_clean', 'Wealth_Idx_Lb', 
                  'urban_flag', 'tv_access']].dropna(subset=['edu_cat', 'Edu_level'])

print(f"Размер выборки для анализа: {len(analysis_df)} записей")

# ============================================
# ШАГ 4: Метод 1 - Сравнение групп
# ============================================

print("\n" + "=" * 80)
print("[4] МЕТОД 1: Сравнение групп")
print("=" * 80)

# --- Анемия детей по образованию матери ---
print("\n--- СВЯЗЬ ОБРАЗОВАНИЯ МАТЕРИ И АНЕМИИ ДЕТЕЙ ---")
ct_anemia = pd.crosstab(analysis_df['edu_cat'], analysis_df['child_anemia'])
pct_anemia = pd.crosstab(analysis_df['edu_cat'], analysis_df['child_anemia'], normalize='index') * 100

print("\nДоля детей с анемией по образованию матери:")
for edu in pct_anemia.index:
    print(f"  {edu}: {pct_anemia.loc[edu, 1]:.1f}%")

chi2, p_val, dof, _ = chi2_contingency(ct_anemia)
print(f"\nX2 = {chi2:.2f}, p-value = {p_val:.4f}")

# --- Вакцинация детей ---
print("\n--- СВЯЗЬ ОБРАЗОВАНИЯ МАТЕРИ И ВАКЦИНАЦИИ ДЕТЕЙ ---")
ct_vacc = pd.crosstab(analysis_df['edu_cat'], analysis_df['measles_vacc'])
pct_vacc = pd.crosstab(analysis_df['edu_cat'], analysis_df['measles_vacc'], normalize='index') * 100

print("\nДоля вакцинированных детей от кори:")
for edu in pct_vacc.index:
    print(f"  {edu}: {pct_vacc.loc[edu, 1]:.1f}%")

chi2, p_val, dof, _ = chi2_contingency(ct_vacc)
print(f"\nX2 = {chi2:.2f}, p-value = {p_val:.4f}")

# --- Вес детей при рождении ---
print("\n--- СВЯЗЬ ОБРАЗОВАНИЯ МАТЕРИ И ВЕСА ДЕТЕЙ ПРИ РОЖДЕНИИ ---")
weight_stats = analysis_df.groupby('edu_cat')['birth_weight_clean'].agg(['mean', 'std', 'count'])
print("\nСредний вес детей по группам образования матери:")
for edu in weight_stats.index:
    print(f"  {edu}: {weight_stats.loc[edu, 'mean']:.2f} ± {weight_stats.loc[edu, 'std']:.2f} кг (n={weight_stats.loc[edu, 'count']:.0f})")

# ANOVA
groups = [analysis_df[analysis_df['edu_cat'] == edu]['birth_weight_clean'].dropna() 
          for edu in analysis_df['edu_cat'].unique()]
f_stat, p_val = f_oneway(*groups)
print(f"\nANOVA: F = {f_stat:.2f}, p-value = {p_val:.4f}")

# --- Уровень гемоглобина детей ---
print("\n--- СВЯЗЬ ОБРАЗОВАНИЯ МАТЕРИ И ГЕМОГЛОБИНА ДЕТЕЙ ---")
hb_stats = analysis_df.groupby('edu_cat')['Hg_child_adjust'].agg(['mean', 'std', 'count'])
print("\nСредний гемоглобин детей по группам образования матери (г/л):")
for edu in hb_stats.index:
    print(f"  {edu}: {hb_stats.loc[edu, 'mean']:.1f} ± {hb_stats.loc[edu, 'std']:.1f} (n={hb_stats.loc[edu, 'count']:.0f})")

# ANOVA для гемоглобина
groups_hb = [analysis_df[analysis_df['edu_cat'] == edu]['Hg_child_adjust'].dropna() 
             for edu in analysis_df['edu_cat'].unique()]
f_stat_hb, p_val_hb = f_oneway(*groups_hb)
print(f"\nANOVA: F = {f_stat_hb:.2f}, p-value = {p_val_hb:.4f}")

# --- Доступ к информации ---
print("\n--- СВЯЗЬ ОБРАЗОВАНИЯ МАТЕРИ И ДОСТУПА К ТВ ---")
tv_by_edu = analysis_df.groupby('edu_cat')['tv_access'].mean() * 100
print("\nДоля семей с телевизором по образованию матери:")
for edu in tv_by_edu.index:
    print(f"  {edu}: {tv_by_edu.loc[edu]:.1f}%")

# ============================================
# ШАГ 5: Метод 2 - Логистическая регрессия
# ============================================

print("\n" + "=" * 80)
print("[5] МЕТОД 2: Логистическая регрессия (анемия детей)")
print("=" * 80)

# Подготовка данных для регрессии
reg_df = analysis_df[['child_anemia', 'Wealth_Idx_Lb', 'Edu_level', 'urban_flag', 'tv_access']].dropna()
y = reg_df['child_anemia']

print(f"\nРазмер выборки для регрессии: {len(reg_df)}")
print(f"Доля детей с анемией: {reg_df['child_anemia'].mean()*100:.1f}%")

# МОДЕЛЬ 1: Только Wealth
X1 = reg_df[['Wealth_Idx_Lb']]
model1 = LogisticRegression(max_iter=1000)
model1.fit(X1, y)
auc1 = roc_auc_score(y, model1.predict_proba(X1)[:, 1])
print(f"\nМОДЕЛЬ 1 (только Wealth):")
print(f"  AUC-ROC = {auc1:.3f}")
print(f"  Коэф. Wealth = {model1.coef_[0][0]:.3f}")

# МОДЕЛЬ 2: Только Education
X2 = reg_df[['Edu_level']]
model2 = LogisticRegression(max_iter=1000)
model2.fit(X2, y)
auc2 = roc_auc_score(y, model2.predict_proba(X2)[:, 1])
print(f"\nМОДЕЛЬ 2 (только Education):")
print(f"  AUC-ROC = {auc2:.3f}")
print(f"  Коэф. Education = {model2.coef_[0][0]:.3f}")

# МОДЕЛЬ 3: Wealth + Education
X3 = reg_df[['Wealth_Idx_Lb', 'Edu_level']]
model3 = LogisticRegression(max_iter=1000)
model3.fit(X3, y)
auc3 = roc_auc_score(y, model3.predict_proba(X3)[:, 1])
print(f"\nМОДЕЛЬ 3 (Wealth + Education):")
print(f"  AUC-ROC = {auc3:.3f}")
print(f"  Улучшение над Wealth: +{(auc3-auc1)*100:.1f}%")
print(f"  Улучшение над Education: +{(auc3-auc2)*100:.1f}%")
print(f"  Коэф. Wealth = {model3.coef_[0][0]:.3f}")
print(f"  Коэф. Education = {model3.coef_[0][1]:.3f}")

# МОДЕЛЬ 4: Полная модель с контролем
X4 = reg_df[['Wealth_Idx_Lb', 'Edu_level', 'urban_flag', 'tv_access']]
model4 = LogisticRegression(max_iter=1000)
model4.fit(X4, y)
auc4 = roc_auc_score(y, model4.predict_proba(X4)[:, 1])
print(f"\nМОДЕЛЬ 4 (полная с контролем):")
print(f"  AUC-ROC = {auc4:.3f}")
print("\n  Коэффициенты:")
for i, col in enumerate(['Wealth_Idx_Lb', 'Edu_level', 'urban_flag', 'tv_access']):
    print(f"    {col}: {model4.coef_[0][i]:.3f}")

# ============================================
# ШАГ 6: Визуализация
# ============================================

print("\n[6] Построение графиков...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# График 1: Анемия детей по образованию матери
ax1 = axes[0, 0]
anemia_by_edu = analysis_df.groupby('edu_cat')['child_anemia'].mean() * 100
anemia_by_edu.sort_index().plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
ax1.set_ylabel('Доля детей с анемией (%)')
ax1.set_title('Анемия детей по образованию матери')
ax1.axhline(y=anemia_by_edu.mean(), color='red', linestyle='--', 
            label=f'Среднее: {anemia_by_edu.mean():.1f}%')
ax1.legend()
ax1.tick_params(axis='x', rotation=30)
ax1.set_xlabel('')

# График 2: Вакцинация детей по образованию матери
ax2 = axes[0, 1]
vacc_by_edu = analysis_df.groupby('edu_cat')['measles_vacc'].mean() * 100
vacc_by_edu.sort_index().plot(kind='bar', ax=ax2, color='lightgreen', edgecolor='black')
ax2.set_ylabel('Доля вакцинированных детей (%)')
ax2.set_title('Вакцинация детей от кори по образованию матери')
ax2.axhline(y=vacc_by_edu.mean(), color='red', linestyle='--',
            label=f'Среднее: {vacc_by_edu.mean():.1f}%')
ax2.legend()
ax2.tick_params(axis='x', rotation=30)
ax2.set_xlabel('')

# График 3: Вес детей при рождении
ax3 = axes[0, 2]
bp_data = [analysis_df[analysis_df['edu_cat'] == edu]['birth_weight_clean'].dropna() 
           for edu in sorted(analysis_df['edu_cat'].unique())]
ax3.boxplot(bp_data, labels=sorted(analysis_df['edu_cat'].unique()))
ax3.set_xlabel('Образование матери')
ax3.set_ylabel('Вес при рождении (кг)')
ax3.set_title('Вес детей при рождении по образованию матери')
ax3.tick_params(axis='x', rotation=30)

# График 4: Доступ к ТВ по образованию матери
ax4 = axes[1, 0]
tv_by_edu = analysis_df.groupby('edu_cat')['tv_access'].mean() * 100
tv_by_edu.sort_index().plot(kind='bar', ax=ax4, color='orange', edgecolor='black')
ax4.set_ylabel('Доля с ТВ (%)')
ax4.set_title('Доступ к телевизору по образованию матери')
ax4.tick_params(axis='x', rotation=30)
ax4.set_xlabel('')

# График 5: Сравнение гемоглобина детей по образованию матери
ax5 = axes[1, 1]
hb_data = [analysis_df[analysis_df['edu_cat'] == edu]['Hg_child_adjust'].dropna() 
           for edu in sorted(analysis_df['edu_cat'].unique())]
ax5.boxplot(hb_data, labels=sorted(analysis_df['edu_cat'].unique()))
ax5.set_xlabel('Образование матери')
ax5.set_ylabel('Гемоглобин детей (г/л)')
ax5.set_title('Уровень гемоглобина детей по образованию матери')
ax5.axhline(y=110, color='red', linestyle='--', label='Порог анемии (110)')
ax5.legend()
ax5.tick_params(axis='x', rotation=30)

# График 6: Сравнение моделей
ax6 = axes[1, 2]
models = ['Только Wealth', 'Только Education', 'Wealth+Edu']
aucs = [auc1, auc2, auc3]
colors = ['lightblue', 'lightgreen', 'coral']
bars = ax6.bar(models, aucs, color=colors, edgecolor='black')
ax6.set_ylabel('AUC-ROC')
ax6.set_title('Сравнение предсказательной силы моделей\n(предсказание анемии детей)')
ax6.set_ylim([0.5, 0.6])
for i, v in enumerate(aucs):
    ax6.text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')
ax6.tick_params(axis='x', rotation=15)

plt.suptitle('Анализ гипотезы 1: Влияние образования матери на здоровье детей', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n(+) 6 графиков отображены на экране")

# ============================================
# ШАГ 7: Сводная таблица
# ============================================

print("\n" + "=" * 80)
print("[7] СВОДНАЯ СТАТИСТИКА")
print("=" * 80)

summary = analysis_df.groupby('edu_cat').agg({
    'child_anemia': 'mean',
    'measles_vacc': 'mean',
    'birth_weight_clean': 'mean',
    'tv_access': 'mean',
    'Hg_child_adjust': 'mean'
})

# Переводим проценты
summary['child_anemia'] = summary['child_anemia'] * 100
summary['measles_vacc'] = summary['measles_vacc'] * 100
summary['tv_access'] = summary['tv_access'] * 100

summary.columns = ['Анемия детей (%)', 'Вакцинация (%)', 'Вес (кг)', 'Доступ к ТВ (%)', 'Гемоглобин (г/л)']

print("\nСводная таблица (средние значения по группам образования матери):")
print("=" * 90)
print(summary.round(1))

# ============================================
# ШАГ 8: Краткие выводы
# ============================================

print("\n" + "=" * 80)
print("[8] КЛЮЧЕВЫЕ ВЫВОДЫ")
print("=" * 80)

print("\n1. Анемия ДЕТЕЙ по образованию матери:")
print(f"   Нет образования: {summary.loc['Нет образования', 'Анемия детей (%)']:.1f}%")
print(f"   Высшее образование: {summary.loc['Высшее', 'Анемия детей (%)']:.1f}%")

print("\n2. Вакцинация детей от кори:")
print(f"   Нет образования: {summary.loc['Нет образования', 'Вакцинация (%)']:.1f}%")
print(f"   Высшее образование: {summary.loc['Высшее', 'Вакцинация (%)']:.1f}%")

print("\n3. Вес детей при рождении:")
print(f"   Нет образования: {summary.loc['Нет образования', 'Вес (кг)']:.2f} кг")
print(f"   Высшее образование: {summary.loc['Высшее', 'Вес (кг)']:.2f} кг")

print("\n4. Гемоглобин детей (средний):")
print(f"   Нет образования: {summary.loc['Нет образования', 'Гемоглобин (г/л)']:.0f} г/л")
print(f"   Высшее образование: {summary.loc['Высшее', 'Гемоглобин (г/л)']:.0f} г/л")

print("\n5. Сравнение влияния образования и дохода на анемию детей:")
print(f"   AUC Wealth: {auc1:.3f}, AUC Education: {auc2:.3f}, AUC Together: {auc3:.3f}")

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)