import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

import scipy.cluster.hierarchy as sch

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split

from pandas.api.types import is_numeric_dtype


# ============================================
# 🌍 Globals
# ============================================


SMALL_PATH = 'data/small.xlsx'
MEDIUM_PATH = 'data/medium.xlsx'
LARGE_PATH = 'data/large.xlsx'

DELETED_SMALL_PATH = 'out/data_with_gaps/small'
DELETED_MEDIUM_PATH = 'out/data_with_gaps/medium'
DELETED_LARGE_PATH = 'out/data_with_gaps/large'

PERCENTS_OF_GAPS = [0.03, 0.05, 0.1, 0.2, 0.3]


# ============================================
# 📄 Dataset class
# ============================================


class Dataset():
    def __init__(self, path: str):
        self.df = self.load_data(path)       
        self.preprocess()
        
        
    #-------------Data Loading & Preprocess------------    
        
    def load_data(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(path)
        except FileNotFoundError:
            print(f"File {path} not exists")
            df = None
        return df
    
    
    def preprocess(self):
        self.df['date-time'] = pd.to_datetime(self.df['date-time'])
        self.df['cards_number'] = self.df['cards_number'].astype('string')
        

    #--------------Analysis & Diagrams---------------
    
    def count(self):
        df_count = pd.DataFrame()
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                df_count[col] = [self.df[col].mean(), self.df[col].median(), list(self.df[col].mode())[:5]]
            elif pd.api.types.is_string_dtype(self.df[col]):
                df_count[col] = [None, None, list(self.df[col].mode())[0]]
                
        return df_count


    def count_all(self):
        df = self.df
        result = {}

        for col in df.columns:
            s = df[col]

            # выбрасываем NaN / NaT, чтобы не мешали подсчёту
            s_no_na = s.dropna()

            # если в колонке вообще нет значений после dropna
            if s_no_na.empty:
                result[col] = [np.nan, np.nan, np.nan]
                continue

            # Числовые признаки — считаем как обычно
            if pd.api.types.is_numeric_dtype(s_no_na):
                mean_val = s_no_na.mean()
                median_val = s_no_na.median()
                mode_series = s_no_na.mode()
                mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan

            # Даты/время — переводим в int64 (наносекунды), считаем, переводим обратно
            elif pd.api.types.is_datetime64_any_dtype(s_no_na):
                # переводим в числа
                numeric = s_no_na.view('int64')

                mean_num = numeric.mean()
                median_num = numeric.median()
                mode_series = numeric.mode()
                mode_num = mode_series.iloc[0] if not mode_series.empty else np.nan

                # переводим обратно в datetime
                mean_val = pd.to_datetime(mean_num)
                median_val = pd.to_datetime(median_num)
                mode_val = pd.to_datetime(mode_num) if not pd.isna(mode_num) else pd.NaT

            else:
                # Строки/категории: факторизуем (строго переводим в числа)
                # factorize возвращает codes и массив уникальных значений
                codes, uniques = pd.factorize(s_no_na, sort=True)
                codes = pd.Series(codes, index=s_no_na.index)

                # считаем среднее/медиану/моду по числовым кодам
                mean_code = int(round(codes.mean()))
                median_code = int(np.median(codes))
                mode_code = int(codes.mode().iloc[0])

                # ограничим индексы на всякий случай
                max_idx = len(uniques) - 1
                mean_code = min(max(mean_code, 0), max_idx)
                median_code = min(max(median_code, 0), max_idx)
                mode_code = min(max(mode_code, 0), max_idx)

                # переводим коды обратно в признаки
                mean_val = uniques[mean_code]
                median_val = uniques[median_code]
                mode_val = uniques[mode_code]

            result[col] = [mean_val, median_val, mode_val]

        df_count = pd.DataFrame(result, index=['mean', 'median', 'mode'])
        return df_count

    
    
    def draw_hist_store(self):
        counts = self.df['store_name'].value_counts()

        plt.figure(figsize=(10, 8))
        plt.barh(counts.index, counts.values)
        plt.xlabel("Number of rows")
        plt.ylabel("Store Name")
        plt.tight_layout()
        plt.show()
        
        
    def draw_hist_date(self):
        self.df['date-time'] = pd.to_datetime(self.df['date-time'])

        df_month = pd.DataFrame()
        df_month['month'] = self.df['date-time'].dt.month

        def get_season(m):
            if m in [12, 1, 2]:
                return "Winter"
            elif m in [3, 4, 5]:
                return "Spring"
            elif m in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        df_month['season'] = df_month['month'].apply(get_season)

        plt.hist(df_month['season'])
        plt.xlabel("Season")
        plt.ylabel("Number of rows")
        plt.show()
        
        
    def draw_hist_coords(self):
        counts = self.df['coordinates'].value_counts()

        plt.figure(figsize=(12,5))
        plt.barh(counts.index, counts.values)
        plt.xlabel("Number of rows")
        plt.ylabel("Coordinates")
        plt.show()
        
        
    def draw_hist_cats(self):
        counts = self.df['categories'].value_counts()

        plt.figure(figsize=(10, 8))
        plt.barh(counts.index, counts.values)
        plt.xlabel("Number of rows")
        plt.ylabel("Category Name")
        plt.tight_layout()
        plt.show()
        
        
    def draw_hist_brand(self):
        counts = self.df['brands'].value_counts()
        
        plt.hist(counts, bins=30)
        plt.xlabel("Times the value occur")
        plt.ylabel("Number of values with this frequency")
        plt.title("Frequency distribution of categories")
        plt.show()
        
        
    def draw_hist_top_brands(self):
        counts = self.df['brands'].value_counts().head(10)
        
        plt.figure(figsize=(10,5))
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=45)
        plt.xlabel("Brands")
        plt.ylabel("Number of rows")
        plt.title("Top 10 Brands")
        plt.show()
        
        
    def draw_hist_bottom_brands(self):
        counts = self.df['brands'].value_counts().tail(10)
        
        plt.figure(figsize=(10,5))
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=45)
        plt.xlabel("Brands")
        plt.ylabel("Number of rows")
        plt.title("Top 10 Brands")
        plt.show()
        
        
    def draw_hist_price(self):
        plt.hist(self.df.loc[self.df['price'] <= 100000, 'price'], bins=50)
        plt.xlabel("Price")
        plt.ylabel("Number of rows")
        plt.show()
        
        
    def count_unique(self, feature: str):
        counts = self.df[feature].value_counts()
        return counts.describe()
    
    
    def draw_hist_num_products(self):
        counts = self.df['number_of_products'].value_counts().sort_index()
        counts.plot(kind='bar')
        plt.xlabel("Number_of_products")
        plt.ylabel("Number of rows")
        plt.xticks(rotation=0)
        plt.show()
        
        
    def analyse_receipt_id(self):
        print(f"Unique number: {self.df['receipt_id'].nunique()}, Number of rows: {len(self.df)}")
        print("Highest number of unique receipt id for stores:")
        print(self.df.groupby("store_name")["receipt_id"].nunique().head())
    
    
    def draw_hist_total_cost(self):
        plt.hist(self.df.loc[self.df['total_cost'] <= 100000, 'total_cost'], bins=50)
        plt.xlabel("Total cost")
        plt.ylabel("Number of rows")
        plt.show()
        
        
    #---------------------Make Gaps-----------------------
        
    def remove_blocks(self, percent: float=0.3, inplace: bool=True) -> pd.DataFrame:
        if inplace:
            df = self.df
        else:
            df = self.df.copy()
        
        rows, cols = df.shape
        total_cells = rows * cols
        target_remove = int(total_cells * percent)
        
        block_sizes = [(2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
        
        removed = 0
        
        while removed < target_remove:
            block_h, block_w = random.choice(block_sizes)
            
            r = random.randint(0, rows - block_h)
            c = random.randint(0, cols - block_w)
            
            for i in range(r, r + block_h):
                for j in range(c, c + block_w):
                    df.iat[i, j] = np.nan
                    
            removed += block_h * block_w
                    
        return df
    
    
    #----------------Restore Functions--------------------
    
    def inpute_groups(self, df: pd.DataFrame | None=None):
        if df is None:
            df = self.df
            
        countable_features = ['price', 'number_of_products', 'total_cost']
        string_features = ['store_name', 'date-time', 'coordinates', 'categories', 'brands', 'cards_number', 'receipt_id']
        
        groups = {
            "store_name": [['coordinates', 'cards_number'], ['cards_number', 'receipt_id'], ['cards_number', 'total_cost'], ['receipt_id', 'total_cost']],
            "date-time": [['store_name', 'receipt_id'], ['store_name', 'cards_number'], ['receipt_id', 'total_cost']],
            "coordinates": [['store_name', 'receipt_id'], ['date-time', 'receipt_id'], ['store_name', 'cards_number'], ['date-time', 'cards_number'], ['date-time', 'total_cost']],
            "categories": [['store_name', 'brands'], ['coordinates', 'brands'], ['store_name', 'price'], ['coordinates', 'price']],
            "brands": [['store_name', 'categories'], ['coordinates', 'categories'], ['store_name', 'price'], ['coordinates', 'price']],
            "price": [['categories', 'brands'], ['store_name', 'brands'], ['store_name', 'categories'], ['coordinates', 'brands'], ['coordinates', 'categories']],
            "cards_number": [['store_name', 'receipt_id'], ['date-time', 'receipt_id'], ['date-time', 'total_cost']],
            "number_of_products": [['store_name', 'receipt_id'], ['date-time', 'receipt_id'], ['store_name', 'total_cost'], ['date-time', 'total_cost'], ['cards_number', 'receipt_id']],
            "receipt_id": [['date-time', 'cards_number'], ['date-time', 'total_cost'], ['store_name', 'date-time'], ['store_name', 'total_cost'], ['cards_number', 'total_cost']],
            "total_cost": [['store_name', 'receipt_id'], ['date-time', 'receipt_id'], ['date-time', 'cards_number']]
        }
                       
                        
        def make_group(df: pd.DataFrame, features: list[str]):
            return df.dropna(subset=features).groupby(features)
        
        for i in range(2):
            for idx, col in df.isna().stack()[lambda x: x].index:
                row = df.loc[idx]
                
                for group in groups[col]:
                
                    if pd.notna(row[group[0]]) and pd.notna(row[group[1]]):
                        if col in string_features:
                            grp = make_group(df, group)
                            
                            s = grp[col].get_group((row[group[0]], row[group[1]])).dropna()
                            
                            if col == "cards_number":
                                s = s.astype(str)
                            
                            if s.empty:
                                val = pd.NA
                            else:
                                val = s.mode()
                                val=val.iloc[0]
                                
                            if col == "cards_number":
                                val = pd.NA if pd.isna(val) else str(val).replace('.0', '')
                            
                            df.at[idx, col] = val
                        else:
                            grp = make_group(df, group)
                            
                            s = grp[col].get_group((row[group[0]], row[group[1]])).dropna()
                            
                            if s.empty:
                                val = pd.NA
                            else:
                                val = s.median()
                            
                            df.at[idx, col] = val
                            
                        break          
        
        for idx, col in df.isna().stack()[lambda x: x].index:
                row = df.loc[idx]
                
                if col in string_features:
                    df.at[idx, col] = df[col].mode()[0]
                else:
                    df.at[idx, col] = df[col].median()
                    
        df['cards_number'] = df['cards_number'].astype('string').str.replace(r"\.0$", "", regex=True)
        
        return df
    
    
    def _build_zet_vectors(self, df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack, csr_matrix
        import numpy as np

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        
        cat_cols = [c for c in df.columns if c not in num_cols]

        if "cards_number" in df.columns:
            df["cards_number"] = df["cards_number"].astype("string")

        def row_to_tokens(row):
            toks = []
            for c in cat_cols:
                val = row[c]
                if pd.isna(val):
                    toks.append(f"{c}=__MISSING__")
                else:
                    toks.append(f"{c}={str(val)}")
            return " ".join(toks)

        texts = df.apply(row_to_tokens, axis=1)

        self._zet_vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+")
        X_tfidf = self._zet_vectorizer.fit_transform(texts)

        if len(num_cols) > 0:
            Z_list = []
            means = {}
            stds = {}

            for c in num_cols:
                col = df[c].astype(float)
                m = col.mean(skipna=True)
                s = col.std(skipna=True)
                
                if not np.isfinite(s) or s == 0:
                    s = 1.0
                z = (col - m) / s
                z = z.fillna(0.0)
                Z_list.append(z.to_numpy().reshape(-1, 1))
                means[c] = float(m)
                stds[c] = float(s)

            X_num = csr_matrix(np.hstack(Z_list))
            X = hstack([X_tfidf, X_num], format="csr")
        else:
            means, stds = {}, {}
            X = X_tfidf

        meta = {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "means": means,
            "stds": stds,
            "row_index": df.index.to_numpy()
        }
        return X, meta


    def impute_zet(self, df: pd.DataFrame | None = None, k: int = 15, target_cols: list[str] | None = None):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import pandas as pd
        from collections import defaultdict

        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()

        if "cards_number" in df.columns:
            df["cards_number"] = df["cards_number"].astype("string")

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in num_cols]

        X, meta = self._build_zet_vectors(df)

        if target_cols is None:
            cols_with_nan = [c for c in df.columns if df[c].isna().any()]
        else:
            cols_with_nan = [c for c in target_cols if c in df.columns and df[c].isna().any()]
        if not cols_with_nan:
            return df
        def weighted_mode(values, weights):
            score = defaultdict(float)
            for v, w in zip(values, weights):
                if pd.isna(v):
                    continue
                score[v] += float(w)
            if not score:
                return pd.NA
            return max(score.items(), key=lambda x: x[1])[0]

        for col in cols_with_nan:
            candidate_mask = df[col].notna().to_numpy()
            candidate_indices = np.where(candidate_mask)[0]
            if candidate_indices.size == 0:
                continue

            X_cand = X[candidate_indices]

            missing_idx = np.where(df[col].isna().to_numpy())[0]
            if missing_idx.size == 0:
                continue

            for i in missing_idx:
                x_i = X[i]
                sims = cosine_similarity(x_i, X_cand).ravel()
                if sims.size == 0:
                    continue
                top_k_idx = np.argpartition(-sims, min(k, sims.size - 1))[:k]
                neigh_sims = sims[top_k_idx]
                neigh_rows = candidate_indices[top_k_idx]

                total_w = float(neigh_sims.sum())
                if total_w <= 1e-12:
                    if col in num_cols:
                        fill_val = df[col].median()
                    else:
                        try:
                            fill_val = df[col].mode(dropna=True).iloc[0]
                        except Exception:
                            fill_val = pd.NA
                    df.iat[i, df.columns.get_loc(col)] = fill_val
                    continue

                if col in num_cols:
                    vals = df.iloc[neigh_rows][col].to_numpy(dtype=float)
                    fill_val = float((vals * neigh_sims).sum() / total_w)
                else:
                    vals = df.iloc[neigh_rows][col].astype("string").to_numpy()
                    fill_val = weighted_mode(vals, neigh_sims)

                    if col == "cards_number":
                        if pd.isna(fill_val):
                            pass
                        else:
                            fill_val = str(fill_val).replace(".0", "")

                df.iat[i, df.columns.get_loc(col)] = fill_val

        if "cards_number" in df.columns:
            df["cards_number"] = df["cards_number"].astype("string").str.replace(r"\.0$", "", regex=True)

        return df
    
    
    #------------Clusterization----------------
    # def hierarchical_clustering(
    #     self,
    #     features: list[str],
    #     n_clusters: int | None = 3,
    #     *,
    #     linkage: str = "ward",                # "ward" | "complete" | "average" | "single"
    #     metric: str = "euclidean",            # игнорируется для "ward" (всегда euclidean)
    #     distance_threshold: float | None = None,
    #     standardize: bool = True,
    #     label_column: str | None = None,
    #     plot: bool = True,
    #     savefig_path: str | None = None,
    # ):
    #     """
    #     Агломеративная иерархическая кластеризация по выбранным числовым признакам.

    #     Параметры:
    #         features           : список колонок DataFrame для кластеризации (числовые).
    #         n_clusters         : число кластеров; игнорируется, если задан distance_threshold.
    #         linkage            : правило объединения кластеров.
    #         metric             : метрика расстояния (для "ward" всегда "euclidean").
    #         distance_threshold : порог высоты дендрограммы вместо фиксированного числа кластеров.
    #         standardize        : стандартизировать ли признаки (StandardScaler).
    #         label_column       : имя выходной колонки с метками кластеров в self.df.
    #         plot               : рисовать дендрограмму.
    #         savefig_path       : путь для сохранения графика (если нужен).

    #     Возвращает:
    #         (labels: pd.Series, model: AgglomerativeClustering)
    #     """
    #     # --- подготовка данных ---
    #     X = self.df[features].dropna()
    #     num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    #     if not num_cols:
    #         raise ValueError("Нужны числовые признаки для кластеризации.")
    #     if len(num_cols) < len(X.columns):
    #         X = X[num_cols]

    #     X_values = X.values
    #     if standardize:
    #         X_values = StandardScaler().fit_transform(X_values)

    #     # --- модель ---
    #     effective_metric = "euclidean" if linkage == "ward" else metric
    #     model = AgglomerativeClustering(
    #         n_clusters=None if distance_threshold is not None else n_clusters,
    #         linkage=linkage,
    #         metric=effective_metric,
    #         distance_threshold=distance_threshold,
    #         compute_distances=distance_threshold is not None,
    #     ).fit(X_values)

    #     labels = pd.Series(model.labels_, index=X.index, name="cluster")

    #     # --- запись меток в df ---
    #     if label_column is None:
    #         suffix = (
    #             f"thr_{distance_threshold}"
    #             if distance_threshold is not None
    #             else f"{n_clusters}"
    #         )
    #         label_column = f"hclust_{linkage}_{suffix}"
    #     self.df[label_column] = pd.Series(index=self.df.index, dtype="Int64")
    #     self.df.loc[labels.index, label_column] = labels.values

    #     # --- визуализация (дендрограмма) ---
    #     if plot:
    #         Z = sch.linkage(X_values, method=linkage, metric=effective_metric)
    #         plt.figure(figsize=(10, 5))
    #         sch.dendrogram(Z, no_labels=True)
    #         plt.title("Hierarchical clustering dendrogram")
    #         plt.xlabel("Objects")
    #         plt.ylabel("Distance")
    #         if savefig_path:
    #             plt.tight_layout()
    #             plt.savefig(savefig_path, dpi=150)
    #         plt.show()

    #     return labels, model


    def hierarchical_clustering(
        self,
        features: list[str],
        n_clusters: int | None = 3,
        *,
        linkage: str = "ward",                # "ward" | "complete" | "average" | "single"
        metric: str = "euclidean",            # игнорируется для "ward" (всегда euclidean)
        distance_threshold: float | None = None,
        standardize: bool = True,
        label_column: str | None = None,      # колонка для меток из AgglomerativeClustering
        # --- НОВОЕ ---
        cut_distance: float | None = None,    # высота отсечения дендрограммы, напр. 120
        cut_label_column: str | None = None,  # колонка для меток при cut_distance
        # -------------
        plot: bool = True,
        savefig_path: str | None = None,
        ):
        """
        Агломеративная иерархическая кластеризация по выбранным числовым признакам.

        Параметры:
            features           : список колонок DataFrame для кластеризации (числовые).
            n_clusters         : число кластеров; игнорируется, если задан distance_threshold.
            linkage            : правило объединения кластеров.
            metric             : метрика расстояния (для "ward" всегда "euclidean").
            distance_threshold : порог высоты в sklearn-кластере вместо фиксированного числа кластеров.
            standardize        : стандартизировать ли признаки (StandardScaler).
            label_column       : имя выходной колонки с метками кластеров (sklearn) в self.df.
            cut_distance       : уровень отсечения дендрограммы (по матрице Z, SciPy).
            cut_label_column   : имя колонки с метками кластеров на уровне cut_distance.
            plot               : рисовать дендрограмму.
            savefig_path       : путь для сохранения графика (если нужен).

        Возвращает:
            (labels: pd.Series, model: AgglomerativeClustering, cut_labels: pd.Series | None)
        """
        # --- подготовка данных ---
        X = self.df[features].dropna()
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

        if not num_cols:
            raise ValueError("Нужны числовые признаки для кластеризации.")
        if len(num_cols) < len(X.columns):
            X = X[num_cols]

        X_values = X.values
        if standardize:
            X_values = StandardScaler().fit_transform(X_values)

        # --- модель (sklearn) ---
        effective_metric = "euclidean" if linkage == "ward" else metric
        model = AgglomerativeClustering(
            n_clusters=None if distance_threshold is not None else n_clusters,
            linkage=linkage,
            metric=effective_metric,
            distance_threshold=distance_threshold,
            compute_distances=distance_threshold is not None,
        ).fit(X_values)

        labels = pd.Series(model.labels_, index=X.index, name="cluster")

        # --- запись меток sklearn-кластера в df ---
        if label_column is None:
            suffix = (
                f"thr_{distance_threshold}"
                if distance_threshold is not None
                else f"{n_clusters}"
            )
            label_column = f"hclust_{linkage}_{suffix}"

        self.df[label_column] = pd.Series(index=self.df.index, dtype="Int64")
        self.df.loc[labels.index, label_column] = labels.values

        # --- SciPy linkage (для дендрограммы + cut_distance) ---
        cut_labels = None      # то, чем вернём кластеры на уровне cut_distance

        if plot or (cut_distance is not None):
            Z = sch.linkage(X_values, method=linkage, metric=effective_metric)

            # дендрограмма
            if plot:
                plt.figure(figsize=(10, 5))
                sch.dendrogram(Z, no_labels=True)
                if distance_threshold is not None:
                    # горизонтальная линия на уровне distance_threshold (если захочешь)
                    plt.axhline(y=distance_threshold, linestyle="--")
                if cut_distance is not None:
                    # горизонтальная линия на уровне cut_distance
                    plt.axhline(y=cut_distance, linestyle=":", linewidth=1)
                plt.title("Hierarchical clustering dendrogram")
                plt.xlabel("Objects")
                plt.ylabel("Distance")
                if savefig_path:
                    plt.tight_layout()
                    plt.savefig(savefig_path, dpi=150)
                plt.show()

            # --- кластеры на уровне cut_distance ---
            if cut_distance is not None:
                flat_clusters = sch.fcluster(
                    Z,
                    t=cut_distance,
                    criterion="distance"
                )
                cut_labels = pd.Series(
                    flat_clusters,
                    index=X.index,
                    name=f"cluster_d{cut_distance:g}",
                )

                if cut_label_column is None:
                    cut_label_column = f"hclust_{linkage}_cut_{cut_distance:g}"

                self.df[cut_label_column] = pd.Series(index=self.df.index, dtype="Int64")
                self.df.loc[cut_labels.index, cut_label_column] = cut_labels.values

        return labels, model, cut_labels


    def hierarchical_clustering_all_features(
        self,
        features: list[str],
        n_clusters: int = 4,
        *,
        linkage: str = "ward",         # "ward" | "complete" | "average" | "single"
        metric: str = "euclidean",     # для "ward" всегда euclidean
        standardize: bool = True,
        label_column: str | None = None,
        plot: bool = True,
        savefig_path: str | None = None,
        top_categories: int = 3,       # сколько самых частых значений по категориальным выводить
    ):
        """
        Иерархическая кластеризация по ВСЕМ указанным признакам (числовым и категориальным).

        Шаги:
        1. Делим признаки на числовые и категориальные.
        2. Числовые (опционально) стандартизируем, категориальные -> OneHotEncoder.
        3. Делаем AgglomerativeClustering по преобразованным данным.
        4. Строим дендрограмму по тем же преобразованным данным.
        5. Для n_clusters (по умолчанию 4) выводим сводку по исходным признакам.

        Результат:
        labels: pd.Series с номерами кластеров
        model : AgglomerativeClustering
        summary: словарь с описанием кластеров
        """

        # --- подготовка данных ---
        if not features:
            raise ValueError("Список features пуст.")

        X = self.df[features].dropna()
        if X.empty:
            raise ValueError("После dropna() не осталось строк для кластеризации.")

        # Разделяем признаки на числовые и категориальные
        num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]

        if not num_cols and not cat_cols:
            raise ValueError("Не удалось найти признаки для кластеризации.")

        # Препроцессинг: числовые + категориальные
        numeric_transformer = StandardScaler() if standardize and num_cols else "passthrough"
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,   # важно: нужен плотный массив для scipy.linkage
        ) if cat_cols else "passthrough"

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ],
            remainder="drop",
        )

        # Преобразуем данные
        X_trans = preprocessor.fit_transform(X)

        # --- модель ---
        effective_metric = "euclidean" if linkage == "ward" else metric

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=effective_metric,
        ).fit(X_trans)

        labels = pd.Series(model.labels_, index=X.index, name="cluster")

        # --- запись меток в df ---
        if label_column is None:
            label_column = f"hclust_all_{linkage}_{n_clusters}"
        self.df[label_column] = pd.Series(index=self.df.index, dtype="Int64")
        self.df.loc[labels.index, label_column] = labels.values

        # --- визуализация (дендрограмма) ---
        if plot:
            # linkage требует на вход уже числовой массив
            Z = sch.linkage(X_trans, method=linkage, metric=effective_metric)

            plt.figure(figsize=(12, 5))
            sch.dendrogram(Z, no_labels=True)
            plt.title(f"Hierarchical clustering dendrogram ({n_clusters} clusters)")
            plt.xlabel("Objects")
            plt.ylabel("Distance")

            # Примерно показим уровень, соответствующий 4 кластерам
            # (для общего понимания, не идеально, но наглядно)
            try:
                # Высота слияния, оставляющая n_clusters кластеров:
                # берём расстояние (колонка 2) на шаге, когда остаётся n_clusters
                # В Z shape = (n_samples-1, 4). Берём элемент с индексом -n_clusters.
                threshold = Z[-n_clusters, 2]
                plt.axhline(y=threshold, linestyle="--")
            except Exception:
                pass

            if savefig_path:
                plt.tight_layout()
                plt.savefig(savefig_path, dpi=150)
            plt.show()

        # --- описание кластеров в ИСХОДНЫХ признаках (а не в OneHot/StandardScaler) ---
        X_with_labels = X.copy()
        X_with_labels["cluster"] = labels

        summary: dict[int, dict] = {}

        for cl in sorted(labels.unique()):
            group = X_with_labels[X_with_labels["cluster"] == cl].drop(columns=["cluster"])
            size = len(group)

            print(f"\n=== Кластер {cl} (n={size}) ===")

            cluster_info: dict[str, dict] = {
                "size": size,
                "numeric": {},
                "categorical": {},
            }

            # Числовые признаки: базовая статистика
            if num_cols:
                print("\nЧИСЛОВЫЕ ПРИЗНАКИ (describe):")
                desc = group[num_cols].describe().T
                print(desc)

                for col in num_cols:
                    cluster_info["numeric"][col] = {
                        "mean": group[col].mean(),
                        "std": group[col].std(),
                        "min": group[col].min(),
                        "max": group[col].max(),
                    }

            # Категориальные признаки: топ значений
            if cat_cols:
                print("\nКАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (top значения):")
                for col in cat_cols:
                    vc = group[col].value_counts(dropna=False)
                    top = vc.head(top_categories)
                    print(f"\n{col}:")
                    print(top)

                    cluster_info["categorical"][col] = top.to_dict()

            summary[int(cl)] = cluster_info

        return labels, model, summary


    
    
    def add_feature_ranking(self, target):
        import numpy as np
        import pandas as pd
        from pandas.api.types import (
            is_object_dtype, is_string_dtype, is_bool_dtype,
            is_numeric_dtype, is_integer_dtype, is_datetime64_any_dtype,
            CategoricalDtype,
        )
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.linear_model import LinearRegression, RidgeClassifier
        from sklearn.exceptions import ConvergenceWarning
        import warnings

        if not isinstance(target, str):
            raise ValueError("target must be a column name (str)")
        if target not in self.df.columns:
            raise ValueError(f"target column '{target}' not found in dataset")

        y = self.df[target]
        X_full = self.df.drop(columns=[target])

        def is_classification(series: pd.Series) -> bool:
            if not is_numeric_dtype(series):
                return True
            nun = series.nunique(dropna=True)
            if is_integer_dtype(series) and nun <= 12:
                return True
            if nun <= 6:
                return True
            return False

        task_is_clf = is_classification(y)
        scoring = "accuracy" if task_is_clf else "r2"
        estimator = RidgeClassifier(alpha=1.0) if task_is_clf else LinearRegression()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        idx = y.dropna().index
        y = y.loc[idx]
        X_full = X_full.loc[idx]

        def build_preprocessor(cols: list[str]) -> ColumnTransformer:
            sub = X_full[cols]
            num_cols = [c for c in sub.columns if is_numeric_dtype(sub[c])]
            cat_cols = [c for c in sub.columns if is_object_dtype(sub[c]) or is_string_dtype(sub[c]) or isinstance(sub[c].dtype, CategoricalDtype) or is_bool_dtype(sub[c])]
            dt_cols = [c for c in sub.columns if is_datetime64_any_dtype(sub[c])]

            def dt_to_parts(X: pd.DataFrame) -> pd.DataFrame:
                X = X.copy()
                out = pd.DataFrame(index=X.index)
                for c in X.columns:
                    s = pd.to_datetime(X[c], errors="coerce")
                    out[f"{c}__year"] = s.dt.year.astype("float64")
                    out[f"{c}__month"] = s.dt.month.astype("float64")
                    out[f"{c}__day"] = s.dt.day.astype("float64")
                    out[f"{c}__dow"] = s.dt.dayofweek.astype("float64")
                return out

            num_tr = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),
            ])
            cat_tr = Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ])
            dt_tr = Pipeline([
                ("parts", FunctionTransformer(dt_to_parts, validate=False)),
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),
            ])
            transformers = []
            if num_cols:
                transformers.append(("num", num_tr, num_cols))
            if cat_cols:
                transformers.append(("cat", cat_tr, cat_cols))
            if dt_cols:
                transformers.append(("dt", dt_tr, dt_cols))
            if not transformers:
                raise ValueError("No usable columns in the current subset")
            return ColumnTransformer(transformers)

        selected: list[str] = []
        remaining = list(X_full.columns)
        results = []
        prev_score = -np.inf

        while remaining:
            best_feat = None
            best_score = -np.inf
            for f in remaining:
                cols = selected + [f]
                pre = build_preprocessor(cols)
                pipe = Pipeline([("prep", pre), ("est", estimator)])
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        score = float(cross_val_score(pipe, X_full[cols], y, cv=cv, scoring=scoring).mean())
                except Exception:
                    score = -np.inf
                if score > best_score:
                    best_score = score
                    best_feat = f
            selected.append(best_feat)
            remaining.remove(best_feat)
            results.append({
                "feature": best_feat,
                "step": len(selected),
                "cv_score_after_add": best_score,
                "delta": (best_score - prev_score) if np.isfinite(prev_score) else np.nan,
            })
            prev_score = best_score

        return pd.DataFrame(results)


    def add_feature_ranking_global(self):

        features = list(self.df.columns)
        agg = {}
        for tgt in features:
            try:
                rank = self.add_feature_ranking(tgt)
            except Exception:
                continue
            if rank is None or len(rank) == 0:
                continue
            r = rank.copy()
            r["delta"] = r["delta"].fillna(0.0)
            for _, row in r.iterrows():
                f = row["feature"]
                d = float(row["delta"]) if pd.notnull(row["delta"]) else 0.0
                st = int(row["step"]) if pd.notnull(row["step"]) else 0
                if f not in agg:
                    agg[f] = {"sum_delta_pos": 0.0, "sum_delta": 0.0, "count": 0, "count_pos": 0, "steps": []}
                agg[f]["sum_delta"] += d
                agg[f]["count"] += 1
                if d > 0:
                    agg[f]["sum_delta_pos"] += d
                    agg[f]["count_pos"] += 1
                agg[f]["steps"].append(st)
        rows = []
        for f, s in agg.items():
            rows.append({
                "feature": f,
                "targets_covered": s["count"],
                "used_with_gain": s["count_pos"],
                "mean_delta": (s["sum_delta"] / s["count"]) if s["count"] else 0.0,
                "mean_delta_pos": (s["sum_delta_pos"] / s["count_pos"]) if s["count_pos"] else 0.0,
                "sum_delta_pos": s["sum_delta_pos"],
                "median_step": float(np.median(s["steps"])) if s["steps"] else np.nan,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out = out.sort_values(["sum_delta_pos", "mean_delta_pos", "used_with_gain"], ascending=[False, False, False]).reset_index(drop=True)
        return out
    
    
    def hierarchical_clustering_all_features_ordinal(
        self,
        features: list[str],
        n_clusters: int = 4,
        *,
        linkage: str = "ward",         # "ward" | "complete" | "average" | "single"
        metric: str = "euclidean",     # для "ward" всегда euclidean
        standardize: bool = True,
        label_column: str | None = None,
        plot: bool = True,
        savefig_path: str | None = None,
        top_categories: int = 3,       # сколько самых частых значений по категориальным выводить
    ):
        """
        Иерархическая кластеризация по ВСЕМ признакам:
        - числовые остаются как есть (можно стандартизировать)
        - категориальные кодируются целыми числами (ordinal encoding)

        Кластеризация идёт по числовой матрице, а для интерпретации
        используются исходные строковые значения.
        """

        if not features:
            raise ValueError("Список features пуст.")

        # Берём только непустые строки
        X = self.df[features].dropna()
        if X.empty:
            raise ValueError("После dropna() не осталось строк для кластеризации.")

        # Разделяем признаки на числовые и категориальные
        num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]

        if not num_cols and not cat_cols:
            raise ValueError("Не найдено признаков для кластеризации.")

        # --- КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ В ЧИСЛА ---
        X_enc = X.copy()
        encoding_maps: dict[str, dict[int, object]] = {}  # col -> {code: original_value}

        for col in cat_cols:
            # factorize даёт код (0..k-1) и массив уникальных значений
            codes, uniques = pd.factorize(X[col], sort=True)
            X_enc[col] = codes.astype(float)   # float, чтобы нормально работал StandardScaler
            encoding_maps[col] = dict(enumerate(uniques))

        # --- МАССИВ ДЛЯ КЛАСТЕРИЗАЦИИ ---
        use_cols = num_cols + cat_cols
        X_values = X_enc[use_cols].values

        if standardize:
            scaler = StandardScaler()
            X_values = scaler.fit_transform(X_values)

        # --- МОДЕЛЬ ---
        effective_metric = "euclidean" if linkage == "ward" else metric

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=effective_metric,
        ).fit(X_values)

        labels = pd.Series(model.labels_, index=X.index, name="cluster")

        # --- ЗАПИСЬ МЕТОК В df ---
        if label_column is None:
            label_column = f"hclust_ord_{linkage}_{n_clusters}"

        self.df[label_column] = pd.Series(index=self.df.index, dtype="Int64")
        self.df.loc[labels.index, label_column] = labels.values

        # --- ДЕНДРОГРАММА (ОСТОРОЖНО: O(n^2) ПО СТРОКАМ) ---
        if plot:
            Z = sch.linkage(X_values, method=linkage, metric=effective_metric)

            plt.figure(figsize=(12, 5))
            sch.dendrogram(Z, no_labels=True)
            plt.title(f"Hierarchical clustering dendrogram ({n_clusters} clusters)")
            plt.xlabel("Objects")
            plt.ylabel("Distance")

            # Примерный уровень для n_clusters
            try:
                threshold = Z[-n_clusters, 2]
                plt.axhline(y=threshold, linestyle="--")
            except Exception:
                pass

            if savefig_path:
                plt.tight_layout()
                plt.savefig(savefig_path, dpi=150)
            plt.show()

        # --- ОПИСАНИЕ КЛАСТЕРОВ В ИСХОДНЫХ ПРИЗНАКАХ ---
        X_with_labels = X.copy()  # ИСХОДНЫЕ значения (строки, числа)
        X_with_labels["cluster"] = labels

        summary: dict[int, dict] = {}

        for cl in sorted(labels.unique()):
            group = X_with_labels[X_with_labels["cluster"] == cl].drop(columns=["cluster"])
            size = len(group)

            print(f"\n=== Кластер {cl} (n={size}) ===")

            cluster_info: dict[str, dict] = {
                "size": size,
                "numeric": {},
                "categorical": {},
            }
            
            # Числовые признаки
            if num_cols:
                print("\nЧИСЛОВЫЕ ПРИЗНАКИ (describe):")
                desc = group[num_cols].describe().T
                print(desc)

                for col in num_cols:
                    cluster_info["numeric"][col] = {
                        "mean": group[col].mean(),
                        "std": group[col].std(),
                        "min": group[col].min(),
                        "max": group[col].max(),
                    }

            # Категориальные признаки
            if cat_cols:
                print("\nКАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (top значения):")
                for col in cat_cols:
                    vc = group[col].value_counts(dropna=False)
                    top = vc.head(top_categories)
                    print(f"\n{col}:")
                    print(top)

                    cluster_info["categorical"][col] = top.to_dict()

            summary[int(cl)] = cluster_info

        # encoding_maps тут на всякий случай — вдруг захочешь явно смотреть коды
        return labels, model, summary, encoding_maps


# ============================================
# ⚙️ Global Functions
# ============================================


def create_data_with_gaps(data: dict[str, Dataset], percents: list[float]):
    for p in percents:
        deleted_small = data["small"].remove_blocks(percent=p, inplace=False)
        deleted_medium = data["medium"].remove_blocks(percent=p, inplace=False)
        deleted_large = data["large"].remove_blocks(percent=p, inplace=False)
        
        deleted_small.to_excel(f"{DELETED_SMALL_PATH}/{int(p*100)}.xlsx", index=False)
        deleted_medium.to_excel(f"{DELETED_MEDIUM_PATH}/{int(p*100)}.xlsx", index=False)
        deleted_large.to_excel(f"{DELETED_LARGE_PATH}/{int(p*100)}.xlsx", index=False)
        
        
def recover_data():
    data = {
        "small": [(Dataset(f"out/data_with_gaps/small/{int(p*100)}.xlsx"), int(p*100)) for p in PERCENTS_OF_GAPS],
        "medium": [(Dataset(f"out/data_with_gaps/medium/{int(p*100)}.xlsx"), int(p*100)) for p in PERCENTS_OF_GAPS],
        "large": [(Dataset(f"out/data_with_gaps/large/{int(p*100)}.xlsx"), int(p*100)) for p in PERCENTS_OF_GAPS],
    }
    
    #-------------Group Algorithm----------------
    for size, datasets in data.items():
        for (dataset, p) in datasets:
            dataset_inputed = dataset.inpute_groups()
            dataset_inputed.to_excel(f"out/recovered_groups/{size}/{p}.xlsx", index=False)
            
    #-------------Zet Algorithm----------------        
    for size, datasets in data.items():
        for (dataset, p) in datasets:
            dataset_inputed = dataset.impute_zet(k=15)
            dataset_inputed.to_excel(f"out/recovered_zet/{size}/{p}.xlsx", index=False)
            
            
def clustering_ward():
    data = Dataset("out/recovered_groups/small/3.xlsx")
    
    labels, model = data.hierarchical_clustering(
        features=["store_name", "date-time", "coordinates", "brands", "price", "cards_number", "number_of_products", "receipt_id", "total_cost"],
        n_clusters=3,          # или distance_threshold=..., тогда n_clusters игнорируется
        linkage="ward",        # "ward" | "complete" | "average" | "single"
        metric="euclidean",    # для non-ward можно задать, напр. "cosine"
        standardize=True,
        label_column="cluster3",
        plot=True,
        savefig_path=None,
    )
    
    
def clustering_chebyshev():
    data = Dataset("out/recovered_groups/small/3.xlsx")
    
    labels, model = data.hierarchical_clustering(
        features=["store_name", "date-time", "coordinates", "brands", "price", "cards_number", "number_of_products", "receipt_id", "total_cost"],
        n_clusters=3,          # или distance_threshold=..., тогда n_clusters игнорируется
        linkage="complete",        # "ward" | "complete" | "average" | "single"
        metric="chebyshev",    # для non-ward можно задать, напр. "cosine"
        standardize=True,
        label_column="cluster3",
        plot=True,
        savefig_path=None,
    )
    
    
def cluster_ward():
    data = Dataset(SMALL_PATH)
    
    labels, model, summary, enc_maps = data.hierarchical_clustering_all_features_ordinal(
        features=["store_name", "date-time", "coordinates","categories", "brands", "price", "cards_number", "number_of_products", "receipt_id", "total_cost"],              # список колонок
        n_clusters=4,
        linkage="ward",
        standardize=True,
        plot=False,                  # ВАЖНО: без дендограммы
    )


        
        
def cluster_chebyshev():
    data = Dataset(SMALL_PATH)
    
    labels, model, summary, enc_maps = data.hierarchical_clustering_all_features_ordinal(
        features=["store_name", "date-time", "coordinates","categories", "brands", "price", "cards_number", "number_of_products", "receipt_id", "total_cost"],              # список колонок
        n_clusters=4,
        linkage="complete",        # "ward" | "complete" | "average" | "single"
        metric="chebyshev",
        standardize=True,
        plot=False,                  # ВАЖНО: без дендограммы
    )


    
    
def Add_method():
    data = Dataset(SMALL_PATH)
    
    rank = data.add_feature_ranking(target="receipt_id")
    
    print(rank)
    
    
def inf_cluster_ward():
    data = Dataset(SMALL_PATH)
    
    labels, model, summary, enc_maps = data.hierarchical_clustering_all_features_ordinal(
        features=["store_name", "brands", "price", "cards_number", "total_cost"],              # список колонок
        n_clusters=4,
        linkage="ward",
        standardize=True,
        plot=True,                  # ВАЖНО: без дендограммы
    )
    
    
def inf_cluster_chebyshev(path):
    data = Dataset(path)
    
    labels, model, summary, enc_maps = data.hierarchical_clustering_all_features_ordinal(
        features=["store_name", "brands", "price", "cards_number", "total_cost"],              # список колонок
        n_clusters=4,
        linkage="complete",        # "ward" | "complete" | "average" | "single"
        metric="chebyshev",
        standardize=True,
        plot=True,                  # ВАЖНО: без дендограммы
    )
    
    
def inf_cluster_ward_recovered(path):
    data = Dataset(path)
    
    labels, model, summary, enc_maps = data.hierarchical_clustering_all_features_ordinal(
        features=["store_name", "brands", "price", "cards_number", "total_cost"],              # список колонок
        n_clusters=4,
        linkage="ward",
        standardize=True,
        plot=True,                  # ВАЖНО: без дендограммы
    )
        
        
# ============================================
# 📥 Loaded Data
# ============================================


data = {
    "small": Dataset(SMALL_PATH),
#     "medium": Dataset(MEDIUM_PATH),
#     "large": Dataset(LARGE_PATH),
}

if __name__ == "__main__":
    # print("Feature, informativity")
    # print("store_name         0.76")
    # print("coordinates        0.71")
    # print("categories         0.65")
    # print("")
    
    Add_method()
