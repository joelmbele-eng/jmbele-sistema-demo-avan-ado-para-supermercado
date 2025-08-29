"""
Demo Avançado de Sistema para Supermercado – PyQt5 + Matplotlib + ARIMA
Versão final: geração de relatório PDF claro para leigos, com:
- Previsão por Produto Individual e por Categoria
- Gráfico com intervalo de confiança (sombreado)
- Tabela de resumo por Mês (quantidade prevista) com cores alternadas nas colunas para facilitar leitura
- Página de explicação em linguagem simples (o que é previsto, como usar)
- Espaço para inserir o logo JMbele (arquivo jmbele_logo.png na mesma pasta)

Como usar:
1) pip install pyqt5 matplotlib pandas statsmodels numpy
2) python demo_supermercado_qt_advanced_report.py

OBS: o app carrega dados de exemplo automaticamente se não carregar CSV.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, QWidget,
    QLabel, QComboBox, QPushButton, QHBoxLayout, QTabWidget, QTableView, QSpinBox,
    QDoubleSpinBox, QCheckBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec

from statsmodels.tsa.arima.model import ARIMA
from PyQt5.QtGui import QPixmap


# ---------------------- Utilidades ----------------------
class PandasModel(QtCore.QAbstractTableModel):
    """Modelo simples para exibir DataFrame no QTableView."""

    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df.copy()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if self._df is None else self._df.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return None
        if role == QtCore.Qt.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if isinstance(val, (float, np.floating)):
                return f"{val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            return str(val)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or self._df is None:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(section)

    def setDataFrame(self, df):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()


def ensure_datetime(df, col="Data"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def group_by_date(df, product=None, category=None):
    t = df.copy()
    t = ensure_datetime(t, "Data")
    if category and category != "(Total)":
        if "Categoria" in t.columns:
            t = t[t["Categoria"] == category]
    if product and product != "(Total)":
        t = t[t["Produto"] == product]
    s = t.groupby("Data")["Vendas"].sum().sort_index()
    return s


def arima_forecast_with_ci(series, steps=30, alpha=0.05):
    """Retorna (mean, lower, upper, fit_object)"""
    series = pd.Series(series).astype(float)
    best_fit = None
    best_aic = math.inf
    orders = [(1, 1, 1), (2, 1, 2), (3, 1, 2)]
    for order in orders:
        try:
            model = ARIMA(series, order=order)
            fit = model.fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
        except Exception:
            continue
    if best_fit is None:
        fc_mean = np.repeat(series.tail(1).values[0], steps)
        fc_low = fc_mean * 0.6
        fc_high = fc_mean * 1.4
        return fc_mean, fc_low, fc_high, None
    pred = best_fit.get_forecast(steps=steps)
    mean = pred.predicted_mean.values
    ci = pred.conf_int(alpha=alpha)
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values
    return mean, lower, upper, best_fit


def detect_alerts(hist_series, fc_mean, fc_low, fc_high, rise_threshold=0.3, drop_threshold=0.3):
    alerts = []
    hist_avg = float(hist_series.tail(30).mean()) if len(hist_series) >= 1 else 0.0
    fc_avg = float(np.mean(fc_mean)) if len(fc_mean) >= 1 else 0.0
    if hist_avg == 0:
        return alerts
    change = (fc_avg - hist_avg) / hist_avg
    if change >= rise_threshold:
        alerts.append(f"ALERTA: previsão indica aumento médio de {change*100:.1f}% em relação aos últimos 30 dias")
    if change <= -drop_threshold:
        alerts.append(f"ALERTA: previsão indica queda média de {abs(change)*100:.1f}% em relação aos últimos 30 dias")
    rel_width = (np.mean(fc_high - fc_low) / (np.mean(fc_mean) + 1e-9))
    if rel_width > 0.6:
        alerts.append("AVISO: alta incerteza na previsão (intervalo de confiança largo)")
    return alerts


def days_to_stockout(df, product, window=14, current_stock=None):
    t = df[df["Produto"] == product]
    t = ensure_datetime(t, "Data")
    daily = t.groupby("Data")["Vendas"].sum().sort_index()
    if len(daily) == 0:
        return np.inf
    recent = daily.tail(window)
    avg = recent.mean() if len(recent) > 0 else daily.mean()
    if avg <= 0:
        return np.inf
    if current_stock is None:
        last = t.sort_values("Data").tail(1)
        if "Estoque" in last.columns and not last.empty:
            current_stock = float(last["Estoque"].values[0])
        else:
            current_stock = 0.0
    return current_stock / avg


def generate_sample_data(days=180):
    np.random.seed(42)
    start = datetime(2025, 1, 1)
    products = ["Arroz 25Kg", "Óleo 1L", "Leite Caixa", "Massa 500g", "Açúcar 1Kg"]
    categorias = {
        "Arroz 25Kg": "Mercearia",
        "Óleo 1L": "Mercearia",
        "Leite Caixa": "Laticinios",
        "Massa 500g": "Mercearia",
        "Açúcar 1Kg": "Mercearia"
    }
    rows = []
    estoque_base = {p: np.random.randint(80, 220) for p in products}
    for d in range(days):
        date = start + timedelta(days=d)
        for p in products:
            trend = 120 + d * 0.1
            season = 20 * math.sin(2 * math.pi * d / 7)
            noise = np.random.normal(0, 15)
            vendas = max(0, int(trend + season + noise + np.random.randint(-10, 10)))
            preco = {
                "Arroz 25Kg": 15000,
                "Óleo 1L": 6500,
                "Leite Caixa": 5000,
                "Massa 500g": 1800,
                "Açúcar 1Kg": 1200,
            }[p]
            estoque_base[p] = max(0, estoque_base[p] + np.random.randint(-5, 8) - vendas // 100)
            rows.append([date.strftime('%Y-%m-%d'), p, categorias[p], vendas, preco, estoque_base[p]])
    df = pd.DataFrame(rows, columns=["Data", "Produto", "Categoria", "Vendas", "Preco", "Estoque"])
    return df


# ---------------------- Canvas Matplotlib ----------------------
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

    def plot_series(self, x, y, label=None):
        self.ax.clear()
        self.ax.plot(x, y, label=label)
        if label:
            self.ax.legend()
        self.ax.set_xlabel("Data")
        self.ax.set_ylabel("Vendas")
        self.ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.draw()

    def plot_series_with_ci(self, x_hist, y_hist, x_fc, y_fc, y_low, y_high):
        self.ax.clear()
        self.ax.plot(x_hist, y_hist, label="Histórico")
        self.ax.plot(x_fc, y_fc, label="Previsão")
        self.ax.fill_between(x_fc, y_low, y_high, alpha=0.25)
        self.ax.legend()
        self.ax.set_xlabel("Data")
        self.ax.set_ylabel("Vendas")
        self.ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.draw()


# ---------------------- Tabs ----------------------
class TabDashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = pd.DataFrame()
        layout = QVBoxLayout(self)

        filt_layout = QHBoxLayout()
        filt_layout.addWidget(QLabel("Categoria:"))
        self.cmbCategoria = QComboBox()
        self.cmbCategoria.addItem("(Total)")
        filt_layout.addWidget(self.cmbCategoria)
        filt_layout.addWidget(QLabel("Produto:"))
        self.cmbProduto = QComboBox()
        self.cmbProduto.addItem("(Total)")
        filt_layout.addWidget(self.cmbProduto)
        self.btnAtualizar = QPushButton("Atualizar")
        filt_layout.addWidget(self.btnAtualizar)
        filt_layout.addStretch(1)
        layout.addLayout(filt_layout)

        self.canvas = MplCanvas()
        layout.addWidget(self.canvas)

        layout.addWidget(QLabel("Top Produtos (por total vendido):"))
        self.tblRanking = QTableView()
        self.rankModel = PandasModel(pd.DataFrame())
        self.tblRanking.setModel(self.rankModel)
        layout.addWidget(self.tblRanking)

        self.btnAtualizar.clicked.connect(self.refresh)

    def setDataFrame(self, df):
        self.df = df.copy()
        self.fill_filters()
        self.refresh()

    def fill_filters(self):
        self.cmbCategoria.blockSignals(True)
        self.cmbProduto.blockSignals(True)
        self.cmbCategoria.clear()
        self.cmbCategoria.addItem("(Total)")
        self.cmbProduto.clear()
        self.cmbProduto.addItem("(Total)")
        if not self.df.empty:
            if "Categoria" in self.df.columns:
                for c in sorted(self.df["Categoria"].dropna().unique()):
                    self.cmbCategoria.addItem(str(c))
            for p in sorted(self.df["Produto"].dropna().unique()):
                self.cmbProduto.addItem(str(p))
        self.cmbCategoria.blockSignals(False)
        self.cmbProduto.blockSignals(False)

    def refresh(self):
        if self.df.empty:
            self.canvas.ax.clear()
            self.canvas.draw()
            self.rankModel.setDataFrame(pd.DataFrame())
            return
        cat = self.cmbCategoria.currentText()
        prod = self.cmbProduto.currentText()
        s = group_by_date(self.df, product=prod, category=cat)
        if len(s) > 0:
            self.canvas.plot_series(s.index, s.values, label=f"{cat} - {prod}")
        rank = (self.df.groupby("Produto")["Vendas"].sum().sort_values(ascending=False).reset_index())
        rank.columns = ["Produto", "Total Vendas"]
        self.rankModel.setDataFrame(rank)


class TabForecast(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = pd.DataFrame()
        self.last_forecast = None
        layout = QVBoxLayout(self)

        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Categoria:"))
        self.cmbCategoria = QComboBox()
        self.cmbCategoria.addItem("(Total)")
        ctl.addWidget(self.cmbCategoria)
        ctl.addWidget(QLabel("Produto:"))
        self.cmbProduto = QComboBox()
        self.cmbProduto.addItem("(Total)")
        ctl.addWidget(self.cmbProduto)
        ctl.addWidget(QLabel("Passos (dias):"))
        self.spnSteps = QSpinBox()
        self.spnSteps.setRange(7, 180)
        self.spnSteps.setValue(30)
        ctl.addWidget(self.spnSteps)
        ctl.addWidget(QLabel("Ajuste % (ex: 10 = +10%):"))
        self.adj = QDoubleSpinBox()
        self.adj.setRange(-90, 500)
        self.adj.setSingleStep(1)
        self.adj.setValue(0)
        ctl.addWidget(self.adj)
        self.chkAutoAlerts = QCheckBox("Gerar alertas automáticos")
        self.chkAutoAlerts.setChecked(True)
        ctl.addWidget(self.chkAutoAlerts)
        self.btnForecast = QPushButton("Gerar Previsão")
        ctl.addWidget(self.btnForecast)
        ctl.addStretch(1)
        layout.addLayout(ctl)

        self.canvas = MplCanvas()
        layout.addWidget(self.canvas)

        self.lblInfo = QLabel()
        layout.addWidget(self.lblInfo)

        self.tblForecast = QTableView()
        self.tblModel = PandasModel(pd.DataFrame())
        self.tblForecast.setModel(self.tblModel)
        layout.addWidget(self.tblForecast)

        self.btnReport = QPushButton("Gerar relatório PDF")
        layout.addWidget(self.btnReport)

        self.btnForecast.clicked.connect(self.make_forecast)
        self.btnReport.clicked.connect(self.generate_pdf_report)

    def setDataFrame(self, df):
        self.df = df.copy()
        self.fill_filters()
        self.canvas.ax.clear()
        self.canvas.draw()
        self.lblInfo.setText("")

    def fill_filters(self):
        self.cmbCategoria.blockSignals(True)
        self.cmbProduto.blockSignals(True)
        self.cmbCategoria.clear()
        self.cmbCategoria.addItem("(Total)")
        self.cmbProduto.clear()
        self.cmbProduto.addItem("(Total)")
        if not self.df.empty:
            if "Categoria" in self.df.columns:
                for c in sorted(self.df["Categoria"].dropna().unique()):
                    self.cmbCategoria.addItem(str(c))
            for p in sorted(self.df["Produto"].dropna().unique()):
                self.cmbProduto.addItem(str(p))
        self.cmbCategoria.blockSignals(False)
        self.cmbProduto.blockSignals(False)

    def make_forecast(self):
        if self.df.empty:
            QMessageBox.information(self, "Dados", "Carregue um CSV primeiro.")
            return
        cat = self.cmbCategoria.currentText()
        prod = self.cmbProduto.currentText()
        steps = int(self.spnSteps.value())
        adj_pct = float(self.adj.value()) / 100.0
        s = group_by_date(self.df, product=prod, category=cat)
        if len(s) < 30:
            QMessageBox.warning(self, "Série curta", "Poucos pontos para prever. Use dados maiores que 30 dias para melhor resultado.")
            return
        mean, low, high, fit = arima_forecast_with_ci(s.values, steps=steps, alpha=0.05)
        # Aplicar ajuste manual
        mean_adj = mean * (1 + adj_pct)
        low_adj = low * (1 + adj_pct)
        high_adj = high * (1 + adj_pct)
        x_hist = s.index
        last_date = x_hist[-1]
        x_fc = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        self.canvas.plot_series_with_ci(x_hist, s.values, x_fc, mean_adj, low_adj, high_adj)
        # Tabela detalhada
        df_out = pd.DataFrame({"Data": x_fc, "Previsao": mean_adj, "Low95": low_adj, "High95": high_adj})
        self.tblModel.setDataFrame(df_out)
        alerts = []
        if self.chkAutoAlerts.isChecked():
            alerts = detect_alerts(s, mean_adj, low_adj, high_adj)
        info = f"Previsão gerada: {len(mean_adj)} dias. Ajuste aplicado: {adj_pct*100:.1f}%"
        if alerts:
            info += " — " + "; ".join(alerts)
        self.lblInfo.setText(info)
        # store last forecast for report
        self.last_forecast = {
            "product": prod,
            "category": cat,
            "x_hist": x_hist,
            "y_hist": s.values,
            "x_fc": x_fc,
            "mean": mean_adj,
            "low": low_adj,
            "high": high_adj,
            "alerts": alerts,
            "fit": fit
        }

    def _aggregate_monthly(self, x_fc, mean, low, high, product_name=None):
        dfm = pd.DataFrame({"Data": pd.to_datetime(x_fc), "Previsao": mean, "Low": low, "High": high})
        dfm['Mes'] = dfm['Data'].dt.to_period('M')
        agg = dfm.groupby('Mes').agg({'Previsao': 'sum', 'Low': 'sum', 'High': 'sum'}).reset_index()
        agg['Mes_str'] = agg['Mes'].dt.strftime('%Y-%m')
        agg = agg[['Mes_str', 'Previsao', 'Low', 'High']]
        agg.columns = ['Mês', 'Quantidade Prevista', 'Low95', 'High95']
        # Adiciona coluna do produto se fornecido
        if product_name is not None:
            agg.insert(0, 'Produto', product_name)
        return agg

    def generate_pdf_report(self):
        if not self.last_forecast:
            QMessageBox.information(self, "Relatório", "Gere a previsão primeiro.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salvar relatório PDF", "relatorio_previsao.pdf", "PDF (*.pdf)")
        if not path:
            return
        lf = self.last_forecast
        try:
            with PdfPages(path) as pdf:
                # --- Página 1: capa / resumo simples ---
                fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 3])

                # Header with logo (if exists)
                ax0 = fig.add_subplot(gs[0])
                ax0.axis('off')
                logo_path = 'jmbele_logo.png'
                if os.path.exists(logo_path):
                    try:
                        img = plt.imread(logo_path)
                        ax0.imshow(img)
                    except Exception:
                        ax0.text(0.01, 0.6, 'JMbele', fontsize=20, weight='bold')
                else:
                    ax0.text(0.01, 0.6, 'JMbele', fontsize=20, weight='bold')
                ax0.text(0.01, 0.1, 'Relatório de Previsão de Vendas', fontsize=14, weight='bold')
                ax0.text(0.01, 0.02, f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Quick summary box
                ax1 = fig.add_subplot(gs[1])
                ax1.axis('off')
                summary_lines = []
                summary_lines.append(f"Produto: {lf['product']}")
                summary_lines.append(f"Categoria: {lf['category']}")
                summary_lines.append(f"Período de previsão: {lf['x_fc'][0].strftime('%Y-%m-%d')} até {lf['x_fc'][-1].strftime('%Y-%m-%d')}")
                # compute totals
                total_pred = float(np.sum(lf['mean']))
                summary_lines.append(f"Quantidade total prevista (próx. {len(lf['mean'])} dias): {int(total_pred):,}")
                if lf['alerts']:
                    summary_lines.append('ALERTAS DETECTADOS:')
                    summary_lines.extend(lf['alerts'])
                else:
                    summary_lines.append('Sem alertas automáticos.')
                ax1.text(0.01, 0.95, ''.join(summary_lines), fontsize=10, verticalalignment='top', family='monospace')

                # Explanation in plain language
                ax2 = fig.add_subplot(gs[2])
                ax2.axis('off')
                # Centralizar explicação na página, garantindo que o texto não ultrapasse os limites da folha
                expl = (
                    "O que este relatório mostra:\n"
                    "- Quantidade prevista: soma das vendas esperadas para cada mês, com base em dados históricos.\n"
                    "- Intervalo de confiança (Low95-High95): margem onde os valores podem variar.\n"
                    "- Use a Quantidade Prevista para planejar compras e evitar rupturas.\n"
                    "- Alertas automáticos indicam mudanças importantes comparadas aos últimos 30 dias.\n"
                    "Como interpretar (exemplo):\n"
                    "Se a quantidade prevista para '2025-09' for 3.000, isso significa que esperamos vender cerca de 3.000 unidades desse produto naquele mês. A coluna Low95 e High95 mostram o intervalo provável."
                )
                # Use bbox para limitar área do texto e garantir que não ultrapasse os limites
                ax2.text(
                    0.5, 0.5, expl,
                    fontsize=10,
                    verticalalignment='center',
                    horizontalalignment='center',
                    wrap=True,
                    family='monospace',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc")
                )
                pdf.savefig(fig)
                plt.close(fig)

                # --- Página 2: gráfico com CI ---
                figg = plt.figure(figsize=(8.27, 5))
                ax = figg.add_subplot(111)
                ax.plot(lf['x_hist'], lf['y_hist'], label='Histórico')
                ax.plot(lf['x_fc'], lf['mean'], label='Previsão')
                ax.fill_between(lf['x_fc'], lf['low'], lf['high'], alpha=0.25)
                ax.set_title('Previsão de Vendas (com intervalo de confiança)')
                ax.set_xlabel('Data')
                ax.set_ylabel('Vendas')
                ax.legend()
                ax.grid(True, alpha=0.3)
                figg.autofmt_xdate()
                pdf.savefig(figg)
                plt.close(figg)

                # --- Página 3: tabela mensal colorida ---
                # Adiciona nome do produto na tabela
                agg = self._aggregate_monthly(lf['x_fc'], lf['mean'], lf['low'], lf['high'], product_name=lf['product'])
                # tabela simples com cores alternadas nas colunas
                fig2 = plt.figure(figsize=(8.27, 11.69))
                ax_table = fig2.add_subplot(111)
                ax_table.axis('off')
                ax_table.set_title('Resumo por Mês - Quantidade Prevista (com intervalo)')

                # build table cell text
                cell_text = []
                for _, row in agg.iterrows():
                    cell_text.append([row['Produto'], row['Mês'], int(row['Quantidade Prevista']), int(row['Low95']), int(row['High95'])])
                col_labels = ['Produto', 'Mês', 'Quantidade Prevista', 'Low95', 'High95']

                table = ax_table.table(cellText=cell_text, colLabels=col_labels, loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)

                # Apply colors: alternate column background colors for readability
                n_rows = len(cell_text)
                n_cols = len(col_labels)
                # colors: header, col1, col2, col3, col4, col5
                header_color = '#40466e'
                col_colors = ['#f0f0f0', '#ffffff', '#f7fbff', '#ffffff', '#f7fbff']
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_facecolor(header_color)
                        cell.set_text_props(color='w', weight='bold')
                    else:
                        if col < len(col_colors):
                            cell.set_facecolor(col_colors[col])
                pdf.savefig(fig2)
                plt.close(fig2)

                # --- Página 4: Tabela detalhada diária (pequena fonte) ---
                # Adiciona nome do produto na tabela detalhada
                df_tab = pd.DataFrame({
                    'Produto': [lf['product']] * len(lf['x_fc']),
                    'Data': lf['x_fc'].astype(str),
                    'Previsao': np.round(lf['mean'], 2),
                    'Low95': np.round(lf['low'], 2),
                    'High95': np.round(lf['high'], 2)
                })
                fig3 = plt.figure(figsize=(8.27, 11.69))
                ax3 = fig3.add_subplot(111)
                ax3.axis('off')
                ax3.set_title('Previsão Diária Detalhada')
                table2 = ax3.table(cellText=df_tab.values, colLabels=df_tab.columns, loc='center')
                table2.auto_set_font_size(False)
                table2.set_fontsize(7)
                table2.scale(1, 1.0)
                pdf.savefig(fig3)
                plt.close(fig3)

            QMessageBox.information(self, "Relatório", f"Relatório salvo em: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Erro PDF", f"Falha ao gerar PDF: {e}")


class TabInventory(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = pd.DataFrame()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Alerta de Estoque – Dias para ruptura (<= 7 dias em destaque)"))
        self.tbl = QTableView()
        self.model = PandasModel(pd.DataFrame())
        self.tbl.setModel(self.model)
        layout.addWidget(self.tbl)

    def setDataFrame(self, df):
        self.df = df.copy()
        self.refresh()

    def refresh(self):
        if self.df.empty:
            self.model.setDataFrame(pd.DataFrame())
            return
        prods = sorted(self.df['Produto'].dropna().unique())
        rows = []
        for p in prods:
            dts = days_to_stockout(self.df, p, window=14)
            last = self.df[self.df['Produto'] == p].sort_values('Data').tail(1)
            est = float(last['Estoque'].values[0]) if not last.empty and 'Estoque' in last.columns else np.nan
            rows.append([p, est, round(dts, 1) if np.isfinite(dts) else None])
        out = pd.DataFrame(rows, columns=['Produto', 'Estoque Atual', 'Dias p/ Ruptura'])
        out.sort_values('Dias p/ Ruptura', inplace=True)
        self.model.setDataFrame(out)


# ---------------------- Janela Principal ----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('JMbele – Sistema Demo Avançado (PyQt5)')
        self.resize(1200, 800)
        self.df = pd.DataFrame()

        # Visual profissional: fundo claro, bordas suaves, fonte moderna
        self.setStyleSheet("""
            QMainWindow {
                background: #f7f9fc;
            }
            QTabWidget::pane {
                border: 1px solid #d0d7e5;
                border-radius: 8px;
                margin: 8px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #e9eef6;
                border: 1px solid #d0d7e5;
                border-radius: 6px;
                padding: 8px 24px;
                font: 13pt 'Segoe UI', 'Arial';
                color: #2d3a4b;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #1a2a3a;
                font-weight: bold;
            }
            QLabel {
                font: 12pt 'Segoe UI', 'Arial';
                color: #2d3a4b;
            }
            QPushButton {
                background: #3b7ddd;
                color: white;
                border-radius: 6px;
                padding: 6px 18px;
                font: 11pt 'Segoe UI', 'Arial';
            }
            QPushButton:hover {
                background: #2359a6;
            }
            QTableView {
                background: #f7f9fc;
                alternate-background-color: #e9eef6;
                gridline-color: #d0d7e5;
                font: 10pt 'Segoe UI', 'Arial';
            }
        """)

        self.tabs = QTabWidget()
        self.tabDash = TabDashboard()
        self.tabFc = TabForecast()
        self.tabInv = TabInventory()
        self.tabs.addTab(self.tabDash, 'Dashboard')
        self.tabs.addTab(self.tabFc, 'Previsão')
        self.tabs.addTab(self.tabInv, 'Estoque')

        central = QWidget()
        lay = QVBoxLayout(central)

        # Logotipo JMbelel.png centralizado no topo
        logo_path = 'jmbelel.png'
        logo_label = QLabel()
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaledToHeight(80, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText('JMbele')
            logo_label.setStyleSheet('font-size: 32pt; font-weight: bold; color: #3b7ddd;')
        lay.addWidget(logo_label)

        self.hint = QLabel('Dica: Arquivo → Criar CSV Exemplo para começar rápido. Use Previsão → Gerar Previsão para criar relatório.')
        self.hint.setStyleSheet('color: #555; padding: 6px; font: 11pt "Segoe UI", "Arial";')
        lay.addWidget(self.hint)
        lay.addWidget(self.tabs)
        self.setCentralWidget(central)

        self._build_menu()

    def _build_menu(self):
        menubar = self.menuBar()
        mfile = menubar.addMenu('Arquivo')
        actOpen = mfile.addAction('Abrir CSV…')
        actOpen.triggered.connect(self.open_csv)
        actSaveSample = mfile.addAction('Criar CSV Exemplo…')
        actSaveSample.triggered.connect(self.save_sample_csv)
        mfile.addSeparator()
        actQuit = mfile.addAction('Sair')
        actQuit.triggered.connect(self.close)

    def set_dataframe_all(self, df):
        self.df = df.copy()
        expected = ['Data', 'Produto', 'Vendas', 'Preco', 'Estoque']
        for c in expected:
            if c not in self.df.columns:
                self.df[c] = np.nan
        self.df = ensure_datetime(self.df, 'Data')
        self.tabDash.setDataFrame(self.df)
        self.tabFc.setDataFrame(self.df)
        self.tabInv.setDataFrame(self.df)

    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Abrir CSV', '', 'CSV (*.csv)')
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self.set_dataframe_all(df)
            QMessageBox.information(self, 'Dados', f'Arquivo carregado: {os.path.basename(path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Erro', f'Falha ao ler CSV: {e}')

    def save_sample_csv(self):
        df = generate_sample_data(days=180)
        path, _ = QFileDialog.getSaveFileName(self, 'Salvar CSV Exemplo', 'exemplo_supermercado.csv', 'CSV (*.csv)')
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            QMessageBox.information(self, 'CSV', f'Exemplo salvo em: {path}')
            self.set_dataframe_all(df)
        except Exception as e:
            QMessageBox.critical(self, 'Erro', f'Falha ao salvar CSV: {e}')


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    # Carrega dados de exemplo automaticamente na primeira execução
    w.set_dataframe_all(generate_sample_data(days=180))
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
