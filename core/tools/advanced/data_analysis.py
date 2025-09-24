"""
數據分析工具
支持CSV、JSON數據分析、統計計算、數據可視化等
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from ..base import BaseTool, ToolResult, ToolMetadata, ToolCategory, tool


@tool(
    category=ToolCategory.ADVANCED,
    name="data_analysis",
    description="數據分析工具，支持CSV、JSON數據的統計分析和可視化",
    tags=["data", "analysis", "statistics", "pandas"]
)
class DataAnalysisTool(BaseTool):
    """數據分析工具"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        
        self.metadata.input_schema = {
            "operation": {"type": "string", "enum": ["load", "describe", "filter", "group", "visualize", "correlation"]},
            "data_source": {"type": "string", "description": "數據源路徑或JSON字符串"},
            "data_type": {"type": "string", "enum": ["csv", "json", "excel"], "default": "csv"},
            "columns": {"type": "array", "description": "選擇的列"},
            "filter_condition": {"type": "string", "description": "過濾條件"},
            "group_by": {"type": "string", "description": "分組列"},
            "agg_function": {"type": "string", "enum": ["mean", "sum", "count", "min", "max"], "default": "mean"},
            "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "histogram", "box"], "default": "bar"}
        }
        
        self.metadata.output_schema = {
            "success": {"type": "boolean"},
            "data": {"type": "object", "description": "處理後的數據"},
            "statistics": {"type": "object", "description": "統計信息"},
            "chart": {"type": "string", "description": "圖表的base64編碼"}
        }
    
    async def execute(self, operation: str, data_source: str = None, data_type: str = "csv",
                     columns: List[str] = None, filter_condition: str = None,
                     group_by: str = None, agg_function: str = "mean",
                     chart_type: str = "bar", **kwargs) -> ToolResult:
        """執行數據分析操作"""
        
        try:
            if operation == "load":
                return await self._load_data(data_source, data_type)
            elif operation == "describe":
                df = await self._get_dataframe(data_source, data_type)
                return await self._describe_data(df, columns)
            elif operation == "filter":
                df = await self._get_dataframe(data_source, data_type)
                return await self._filter_data(df, filter_condition, columns)
            elif operation == "group":
                df = await self._get_dataframe(data_source, data_type)
                return await self._group_data(df, group_by, agg_function, columns)
            elif operation == "visualize":
                df = await self._get_dataframe(data_source, data_type)
                return await self._visualize_data(df, chart_type, columns)
            elif operation == "correlation":
                df = await self._get_dataframe(data_source, data_type)
                return await self._correlation_analysis(df, columns)
            else:
                return ToolResult(False, error=f"Unsupported operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, error=f"Data analysis failed: {str(e)}")
    
    async def _get_dataframe(self, data_source: str, data_type: str) -> pd.DataFrame:
        """獲取DataFrame"""
        if data_type == "csv":
            return pd.read_csv(data_source)
        elif data_type == "json":
            if Path(data_source).exists():
                return pd.read_json(data_source)
            else:
                # 嘗試解析為JSON字符串
                data = json.loads(data_source)
                return pd.DataFrame(data)
        elif data_type == "excel":
            return pd.read_excel(data_source)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    async def _load_data(self, data_source: str, data_type: str) -> ToolResult:
        """加載數據"""
        try:
            df = await self._get_dataframe(data_source, data_type)
            
            data_info = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            # 預覽數據
            preview = df.head().to_dict('records')
            
            result = ToolResult(True, data={
                "data_info": data_info,
                "preview": preview,
                "message": f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns"
            })
            
            # 分享數據
            result.add_shareable_data("dataframe_info", data_info)
            result.add_shareable_data("data_source", data_source)
            result.add_shareable_data("data_type", data_type)
            
            # 建議下一步操作
            result.suggest_next_tool("data_analysis", "數據已加載，建議進行描述性統計分析")
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to load data: {str(e)}")
    
    async def _describe_data(self, df: pd.DataFrame, columns: List[str] = None) -> ToolResult:
        """描述性統計分析"""
        try:
            if columns:
                df = df[columns]
            
            # 基本統計
            description = df.describe(include='all').to_dict()
            
            # 數值型列的額外統計
            numeric_stats = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                numeric_stats[col] = {
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                    "variance": df[col].var()
                }
            
            # 類別型列的統計
            categorical_stats = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                categorical_stats[col] = {
                    "unique_count": df[col].nunique(),
                    "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "value_counts": df[col].value_counts().head().to_dict()
                }
            
            result = ToolResult(True, data={
                "basic_statistics": description,
                "numeric_statistics": numeric_stats,
                "categorical_statistics": categorical_stats,
                "shape": df.shape
            })
            
            result.add_shareable_data("statistics", {
                "basic": description,
                "numeric": numeric_stats,
                "categorical": categorical_stats
            })
            
            result.suggest_next_tool("data_visualization", "統計分析完成，建議創建可視化圖表")
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to describe data: {str(e)}")
    
    async def _filter_data(self, df: pd.DataFrame, filter_condition: str, 
                          columns: List[str] = None) -> ToolResult:
        """過濾數據"""
        try:
            # 執行過濾條件
            filtered_df = df.query(filter_condition) if filter_condition else df
            
            if columns:
                filtered_df = filtered_df[columns]
            
            result_data = {
                "filtered_data": filtered_df.head(100).to_dict('records'),  # 限制返回數量
                "original_shape": df.shape,
                "filtered_shape": filtered_df.shape,
                "filter_condition": filter_condition
            }
            
            result = ToolResult(True, data=result_data)
            
            result.add_shareable_data("filtered_data", filtered_df.to_dict('records'))
            result.add_shareable_data("filter_applied", filter_condition)
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to filter data: {str(e)}")
    
    async def _group_data(self, df: pd.DataFrame, group_by: str, 
                         agg_function: str, columns: List[str] = None) -> ToolResult:
        """分組數據"""
        try:
            if not group_by or group_by not in df.columns:
                return ToolResult(False, error=f"Invalid group_by column: {group_by}")
            
            # 選擇數值型列進行聚合
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if columns:
                numeric_columns = [col for col in columns if col in numeric_columns]
            
            if not numeric_columns:
                return ToolResult(False, error="No numeric columns found for aggregation")
            
            # 執行分組聚合
            grouped = df.groupby(group_by)[numeric_columns]
            
            if agg_function == "mean":
                result_df = grouped.mean()
            elif agg_function == "sum":
                result_df = grouped.sum()
            elif agg_function == "count":
                result_df = grouped.count()
            elif agg_function == "min":
                result_df = grouped.min()
            elif agg_function == "max":
                result_df = grouped.max()
            else:
                return ToolResult(False, error=f"Unsupported aggregation function: {agg_function}")
            
            grouped_data = result_df.to_dict('index')
            
            result = ToolResult(True, data={
                "grouped_data": grouped_data,
                "group_by": group_by,
                "aggregation": agg_function,
                "columns_aggregated": numeric_columns
            })
            
            result.add_shareable_data("grouped_data", grouped_data)
            result.add_shareable_data("group_analysis", {
                "group_by": group_by,
                "function": agg_function
            })
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to group data: {str(e)}")
    
    async def _visualize_data(self, df: pd.DataFrame, chart_type: str, 
                             columns: List[str] = None) -> ToolResult:
        """數據可視化"""
        try:
            plt.figure(figsize=(10, 6))
            
            if columns and len(columns) >= 1:
                if chart_type == "bar":
                    if len(columns) == 1:
                        df[columns[0]].value_counts().plot(kind='bar')
                    else:
                        df[columns].plot(kind='bar')
                elif chart_type == "line":
                    df[columns].plot(kind='line')
                elif chart_type == "scatter":
                    if len(columns) >= 2:
                        plt.scatter(df[columns[0]], df[columns[1]])
                        plt.xlabel(columns[0])
                        plt.ylabel(columns[1])
                elif chart_type == "histogram":
                    df[columns[0]].hist(bins=30)
                elif chart_type == "box":
                    df[columns].boxplot()
            else:
                # 使用數值型列
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
                if len(numeric_cols) > 0:
                    df[numeric_cols].plot(kind=chart_type)
            
            plt.title(f"{chart_type.capitalize()} Chart")
            plt.tight_layout()
            
            # 將圖表轉換為base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            plt.close()
            
            result = ToolResult(True, data={
                "chart_base64": image_base64,
                "chart_type": chart_type,
                "columns_used": columns or numeric_cols.tolist()
            })
            
            result.add_shareable_data("chart", image_base64)
            result.add_shareable_data("visualization_type", chart_type)
            
            return result
            
        except Exception as e:
            plt.close()  # 確保清理
            return ToolResult(False, error=f"Failed to create visualization: {str(e)}")
    
    async def _correlation_analysis(self, df: pd.DataFrame, columns: List[str] = None) -> ToolResult:
        """相關性分析"""
        try:
            # 選擇數值型列
            numeric_df = df.select_dtypes(include=[np.number])
            if columns:
                numeric_df = numeric_df[[col for col in columns if col in numeric_df.columns]]
            
            if numeric_df.empty:
                return ToolResult(False, error="No numeric columns found for correlation analysis")
            
            # 計算相關係數
            correlation_matrix = numeric_df.corr()
            
            # 創建熱力圖
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            # 轉換為base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            plt.close()
            
            # 找出強相關關係
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # 強相關閾值
                        strong_correlations.append({
                            "variable1": correlation_matrix.columns[i],
                            "variable2": correlation_matrix.columns[j],
                            "correlation": corr_value
                        })
            
            result = ToolResult(True, data={
                "correlation_matrix": correlation_matrix.to_dict(),
                "strong_correlations": strong_correlations,
                "heatmap_base64": heatmap_base64
            })
            
            result.add_shareable_data("correlations", correlation_matrix.to_dict())
            result.add_shareable_data("strong_correlations", strong_correlations)
            
            return result
            
        except Exception as e:
            plt.close()
            return ToolResult(False, error=f"Failed to perform correlation analysis: {str(e)}")


@tool(
    category=ToolCategory.ADVANCED,
    name="web_scraping",
    description="網頁抓取工具，支持HTML內容提取、數據爬取、API調用等",
    tags=["web", "scraping", "requests", "html"]
)
class WebScrapingTool(BaseTool):
    """網頁抓取工具"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        
        self.metadata.input_schema = {
            "operation": {"type": "string", "enum": ["get", "post", "extract", "api_call"]},
            "url": {"type": "string", "description": "目標URL"},
            "headers": {"type": "object", "description": "HTTP頭"},
            "params": {"type": "object", "description": "請求參數"},
            "data": {"type": "object", "description": "POST數據"},
            "selector": {"type": "string", "description": "CSS選擇器"},
            "timeout": {"type": "number", "default": 30}
        }
    
    async def execute(self, operation: str, url: str, headers: Dict = None,
                     params: Dict = None, data: Dict = None, selector: str = None,
                     timeout: int = 30, **kwargs) -> ToolResult:
        """執行網頁抓取操作"""
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            if operation == "get":
                return await self._get_request(url, headers, params, timeout)
            elif operation == "post":
                return await self._post_request(url, headers, data, timeout)
            elif operation == "extract":
                return await self._extract_content(url, selector, headers, timeout)
            elif operation == "api_call":
                return await self._api_call(url, headers, params, data, timeout)
            else:
                return ToolResult(False, error=f"Unsupported operation: {operation}")
                
        except ImportError:
            return ToolResult(False, error="Required packages not installed: requests, beautifulsoup4")
        except Exception as e:
            return ToolResult(False, error=f"Web scraping failed: {str(e)}")
    
    async def _get_request(self, url: str, headers: Dict, params: Dict, timeout: int) -> ToolResult:
        """GET請求"""
        import requests
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            
            result_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:10000],  # 限制內容長度
                "url": response.url,
                "encoding": response.encoding
            }
            
            result = ToolResult(True, data=result_data)
            result.add_shareable_data("web_content", response.text)
            result.add_shareable_data("response_headers", dict(response.headers))
            
            result.suggest_next_tool("text_analysis", "網頁內容已獲取，建議進行文本分析")
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"GET request failed: {str(e)}")
    
    async def _post_request(self, url: str, headers: Dict, data: Dict, timeout: int) -> ToolResult:
        """POST請求"""
        import requests
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            response.raise_for_status()
            
            result_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:10000],
                "url": response.url
            }
            
            result = ToolResult(True, data=result_data)
            result.add_shareable_data("api_response", response.text)
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"POST request failed: {str(e)}")
    
    async def _extract_content(self, url: str, selector: str, headers: Dict, timeout: int) -> ToolResult:
        """提取網頁內容"""
        import requests
        from bs4 import BeautifulSoup
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if selector:
                elements = soup.select(selector)
                extracted_content = [elem.get_text(strip=True) for elem in elements]
            else:
                # 提取常見內容
                extracted_content = {
                    "title": soup.title.string if soup.title else None,
                    "headings": [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])],
                    "paragraphs": [p.get_text(strip=True) for p in soup.find_all('p')[:10]],
                    "links": [{"text": a.get_text(strip=True), "href": a.get('href')} 
                             for a in soup.find_all('a', href=True)[:20]]
                }
            
            result = ToolResult(True, data={
                "extracted_content": extracted_content,
                "selector_used": selector,
                "url": url
            })
            
            result.add_shareable_data("extracted_data", extracted_content)
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Content extraction failed: {str(e)}")
    
    async def _api_call(self, url: str, headers: Dict, params: Dict, data: Dict, timeout: int) -> ToolResult:
        """API調用"""
        import requests
        
        try:
            if data:
                response = requests.post(url, headers=headers, params=params, json=data, timeout=timeout)
            else:
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            
            response.raise_for_status()
            
            try:
                json_data = response.json()
            except:
                json_data = None
            
            result_data = {
                "status_code": response.status_code,
                "json_data": json_data,
                "text_data": response.text if not json_data else None,
                "headers": dict(response.headers)
            }
            
            result = ToolResult(True, data=result_data)
            result.add_shareable_data("api_data", json_data or response.text)
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"API call failed: {str(e)}")

