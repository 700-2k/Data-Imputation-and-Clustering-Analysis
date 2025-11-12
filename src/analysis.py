import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


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
        
        
def recover_data_groups():
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
        
        
# ============================================
# 📥 Loaded Data
# ============================================


# data = {
#     "small": Dataset(SMALL_PATH),
#     "medium": Dataset(MEDIUM_PATH),
#     "large": Dataset(LARGE_PATH),
# }

if __name__ == "__main__":
    recover_data_groups()
