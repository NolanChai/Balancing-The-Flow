import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.font_manager
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, LogFormatter
from matplotlib.patches import Patch
from matplotlib import rc
import warnings
import os
from pathlib import Path
from functools import partial
import datetime

def log_time():
    now = datetime.datetime.now()
    log = f"[{now:%Y-%m-%d %H:%M:%S}]"
    return log

class UIDPlotter:
    """
    Advanced plotting system for UID metrics with publication-quality aesthetics and layout flexibility.
    Supports composable plots, grid arrangements, and multiple visualization types.
    """
    def __init__(self, use_tex=True, style="seaborn-v0_8-whitegrid", font_family="serif", font_serif="Times"):
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Set style
        plt.style.use(style)
        
        # Configure plot settings with LaTeX if requested
        if use_tex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": font_family,
                "font.serif": font_serif,
            })
        
        # Define color schemes for different model types (customizable)
        self.colors = {
            "gpt2": "#303030",
            "llama-2": "#003f5c",
            "mistral": "#58508d",
            "phi": "#bc5090",
            "claude": "#ff6361",
            "gemini": "#ffa600",
            # Add more models as needed
        }
        
        # Alternative color palettes
        self.palettes = {
            "default": self.colors,
            "colorblind": sns.color_palette("colorblind", n_colors=10),
            "pastel": sns.color_palette("pastel", n_colors=10),
            "deep": sns.color_palette("deep", n_colors=10),
            "muted": sns.color_palette("muted", n_colors=10),
            "bright": sns.color_palette("bright", n_colors=10),
            "dark": sns.color_palette("dark", n_colors=10)
        }
        
        # Set model order for consistent plotting
        self.model_order = ["gpt2", "llama-2", "mistral", "phi", "claude", "gemini"]
        
        # Define markers for different models
        self.markers = {
            "gpt2": "o",
            "llama-2": "s",
            "mistral": "h",
            "phi": "^",
            "claude": "X",
            "gemini": "*",
        }
        
        # Set default sizes
        self.HEIGHT = 1.6
        self.ASPECT = 0.8
        self.FONTSIZE = 18
        
        # Define model name mappings for nicer labels
        self.rename_model = {
            "gpt2": "GPT-2",
            "llama-2": "Llama-2",
            "mistral": "Mistral",
            "phi": "Phi",
            "claude": "Claude",
            "gemini": "Gemini"
        }
        
        # Dataset name mappings
        self.rename_dataset = {
            "cnn_dailymail": "CNN/Daily Mail",
            "daily_dialog": "Daily Dialog"
        }
        
        # Metric name mappings for nicer labels
        self.rename_metric = {
            "uid_variance": "UID Variance",
            "uid_pairwise": "UID Pairwise",
            "mean_surprisal": "Mean Surprisal",
            "tokens": "Token Count",
            "vocab_size": "Vocabulary Size",
            "sentence_length": "Sentence Length"
        }
        
        # Set default style parameters
        self.style_params = {
            "grid_color": "0.8",
            "spine_color": "black",
            "tick_color": "black",
            "alpha": 0.7,
            "linewidth": 1.0,
            "markersize": 6,
            "use_color_by_model": True,
            "default_palette": "default",
            "figure_dpi": 300
        }
        
    def customize(self, **kwargs):
        """
        Customize plotting parameters.
        
        Args:
            **kwargs: Parameters to customize (colors, models, rename_model, HEIGHT, ASPECT, etc.)
        """
        for key, value in kwargs.items():
            if key == "colors" and isinstance(value, dict):
                self.colors.update(value)
                self.palettes["default"] = self.colors
            elif key == "rename_model" and isinstance(value, dict):
                self.rename_model.update(value)
            elif key == "rename_dataset" and isinstance(value, dict):
                self.rename_dataset.update(value)
            elif key == "rename_metric" and isinstance(value, dict):
                self.rename_metric.update(value)
            elif key == "model_order" and isinstance(value, list):
                self.model_order = value
            elif key == "markers" and isinstance(value, dict):
                self.markers.update(value)
            elif key == "style_params" and isinstance(value, dict):
                self.style_params.update(value)
            elif hasattr(self, key):
                setattr(self, key, value)
                
        return self

    def _set_theme(self, ax):
        """Apply consistent theme to a matplotlib axis"""
        # Grid and spines
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color(self.style_params['spine_color'])
        ax.spines['right'].set_visible(True)
        ax.spines['right'].set_color(self.style_params['spine_color'])
        
        # Set grid parameters
        ax.grid(color=self.style_params['grid_color'], zorder=-1)
        
        # Set zorder for patches
        for patch in ax.patches:
            patch.set_zorder(2)
    
    def _apply_theme_to_facetgrid(self, facetgrid):
        """Apply theme to all axes in a facetgrid"""
        if hasattr(facetgrid, 'axes'):
            axes = facetgrid.axes.flat if hasattr(facetgrid.axes, 'flat') else [facetgrid.axes]
            for ax in axes:
                self._set_theme(ax)

    def _fix_labels(self, g, xlabel="", ylabel="", fontsize=None, x_formatter=None, y_formatter="2_sig_figs", 
                   rotate_xlabels=45):
        """Fix labels and formatting in the facetgrid or regular plot"""
        fontsize = fontsize or self.FONTSIZE
        
        if hasattr(g, 'fig'):
            g.fig.canvas.draw()
            
        # Handle different types of plot objects
        if hasattr(g, 'axes') and hasattr(g.axes, 'shape'):
            rows, cols = g.axes.shape
            axes = g.axes.flat
        elif hasattr(g, 'axes') and not hasattr(g.axes, 'shape'):
            if isinstance(g.axes, np.ndarray):
                rows, cols = 1, len(g.axes)
                axes = g.axes
            else:
                rows, cols = 1, 1
                axes = [g.axes]
        else:
            # Handle case where g is a matplotlib figure
            axes = g.get_axes() if hasattr(g, 'get_axes') else [plt.gca()]
            rows, cols = 1, len(axes)
        
        for i, ax in enumerate(axes):
            r = i // cols if cols > 0 else 0
            c = i % cols if cols > 0 else i
            
            # Set x-tick formatting
            if x_formatter == "int":
                ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
                plt.setp(ax.get_xticklabels(), fontsize=fontsize-2)
            else:
                plt.setp(ax.get_xticklabels(), rotation=rotate_xlabels, ha='right', fontsize=fontsize-2)
            
            # Set y-tick formatting
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            if y_formatter == "2_sig_figs":
                yticklabels = [f'{y:.2g}' for y in ax.get_yticks()]
            elif y_formatter == "int":
                yticklabels = [f'{int(y)}' for y in ax.get_yticks()]
            elif y_formatter == "percent":
                yticklabels = [f'{y:.0%}' for y in ax.get_yticks()]
            else:
                yticklabels = [f'{y:.2f}' for y in ax.get_yticks()]
                
            ax.set_yticklabels(yticklabels, fontsize=fontsize-2)
            
            # Set column title if first row
            if r == 0 and cols > 1 and hasattr(ax, 'get_title'):
                title = ax.get_title()
                if title and "|" in title:
                    parts = title.split("|")
                    if len(parts) > 1:
                        col_val = parts[1].strip().split("=")[-1].strip()
                        if col_val in self.rename_dataset:
                            col_val = self.rename_dataset[col_val]
                        ax.text(0.5, 1.1, col_val, transform=ax.transAxes, ha='center', va='center', fontsize=fontsize-2)
            
            # Set x-label if last row
            if rows > 1 and r == rows - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
            elif rows == 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
            
            # Set row title if last column
            if cols > 1 and c == cols - 1 and hasattr(ax, 'get_title'):
                title = ax.get_title()
                if title and "|" in title:
                    parts = title.split("|")
                    if len(parts) > 0:
                        row_val = parts[0].strip().split("=")[-1].strip()
                        if row_val in self.rename_model:
                            row_val = self.rename_model[row_val]
                        ax.text(1.1, 0.5, row_val, transform=ax.transAxes, rotation=270, 
                              ha='center', va='center', fontsize=fontsize)
            
            # Set y-label if first column
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=fontsize)
            
            # Remove the default title
            if hasattr(ax, 'set_title'):
                ax.set_title("")
                
            # Add final theme elements
            self._set_theme(ax)

    def _fix_legend(self, g, coords=(0.88, 0.14), title=None, loc="lower right", frameon=False):
        """Fix legend appearance and position"""
        if hasattr(g, '_legend') and g._legend:
            g._legend.remove()
        
        if hasattr(g, 'axes') and hasattr(g.axes, 'flat'):
            ax = g.axes.flat[0]
        elif hasattr(g, 'axes'):
            ax = g.axes
        else:
            ax = plt.gca()
            
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            if hasattr(g, 'fig'):
                fig = g.fig
            else:
                fig = plt.gcf()
                
            legend = fig.legend(handles=handles, labels=labels, title=title, 
                            bbox_to_anchor=coords, loc=loc, frameon=frameon)
            
            # Adjust legend title font size
            if title and hasattr(legend, 'get_title'):
                legend.get_title().set_fontsize(self.FONTSIZE - 2)
        
    def _remove_legend(self, g):
        """Remove the legend"""
        if hasattr(g, '_legend') and g._legend:
            g._legend.remove()

    def _prepare_data(self, data, model_column='model', dataset_column='dataset'):
        """
        Prepare data for plotting by standardizing model names, etc.
        
        Args:
            data: DataFrame to prepare
            model_column: Name of model column
            dataset_column: Name of dataset column
            
        Returns:
            Prepared DataFrame
        """
        df = data.copy()
        
        # Standardize model names
        if model_column in df.columns:
            # Extract model name from full model name (e.g., llama-2-7b -> llama-2)
            df[model_column] = df[model_column].apply(
                lambda x: next((m for m in self.model_order if m in str(x).lower()), str(x))
            )
            
            # Create a display name column
            df['model_display'] = df[model_column].map(
                lambda x: self.rename_model.get(x, x)
            )
        
        # Standardize dataset names
        if dataset_column in df.columns:
            df['dataset_display'] = df[dataset_column].map(
                lambda x: self.rename_dataset.get(x, x)
            )
            
        return df

    def _get_color_palette(self, df, hue_column, palette_name="default"):
        """Get color palette appropriate for the data"""
        if palette_name == "default" and hue_column == "model" and self.style_params["use_color_by_model"]:
            # Use model-specific colors
            palette = {model: self.colors.get(model, "#333333") 
                    for model in df[hue_column].unique()}
            return palette
        else:
            # Use a named palette
            palette = self.palettes.get(palette_name, "deep")
            return palette

    def create_figure_grid(self, nrows=1, ncols=1, height=None, aspect=None, sharex=False, sharey=False):
        """
        Create a figure with a grid of subplots.
        
        Args:
            nrows: Number of rows
            ncols: Number of columns
            height: Height of each subplot (in inches)
            aspect: Aspect ratio of each subplot (width/height)
            sharex: Whether to share x-axes
            sharey: Whether to share y-axes
            
        Returns:
            fig, axes (where axes is a 2D array-like of axes)
        """
        height = height or self.HEIGHT
        aspect = aspect or self.ASPECT
        
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(ncols * height * aspect, nrows * height),
            sharex=sharex, sharey=sharey,
            squeeze=False  # Always return 2D array of axes
        )
        
        return fig, axes
    
    def create_figure_mosaic(self, layout, height=None, width=None):
        """
        Create a figure with a custom mosaic layout.
        
        Args:
            layout: Mosaic layout specification (list of lists of strings)
            height: Total height of figure (in inches)
            width: Total width of figure (in inches)
            
        Returns:
            fig, axes_dict (where axes_dict maps layout keys to axes)
        """
        height = height or self.HEIGHT * len(layout)
        width = width or self.ASPECT * self.HEIGHT * max(len(row) for row in layout)
        
        fig, axes_dict = plt.subplot_mosaic(
            layout,
            figsize=(width, height)
        )
        
        return fig, axes_dict
        
    def plot_bar(self, data, x, y, hue=None, ax=None, palette=None, order=None, 
                title=None, xlabel=None, ylabel=None, fontsize=None, **kwargs):
        """
        Create a bar plot.
        
        Args:
            data: DataFrame with data
            x: Column for x-axis
            y: Column for y-axis
            hue: Column for color grouping
            ax: Matplotlib axis to plot on
            palette: Color palette to use
            order: Order of x-axis categories
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            fontsize: Font size for labels
            **kwargs: Additional keyword arguments for sns.barplot
            
        Returns:
            The matplotlib axis with the plot
        """
        fontsize = fontsize or self.FONTSIZE
        
        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(self.HEIGHT * self.ASPECT, self.HEIGHT))
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Set default order for x-axis if it's a model column
        if order is None and x == 'model':
            order = [m for m in self.model_order if m in df[x].unique()]
        
        # Get color palette
        if palette is None and hue is not None:
            palette = self._get_color_palette(df, hue)
            
        # Create bar plot
        sns.barplot(
            data=df, x=x, y=y, hue=hue, ax=ax,
            palette=palette, order=order,
            **kwargs
        )
        
        # Add value labels on top of bars
        for i, bar in enumerate(ax.patches):
            value = bar.get_height()
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.02 * ax.get_ylim()[1],
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=fontsize-4
                )
        
        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize)
            
        # Apply theme
        self._set_theme(ax)
        
        # Format tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize-2)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
        
        return ax
    
    def plot_box(self, data, x, y, hue=None, ax=None, palette=None, order=None,
                title=None, xlabel=None, ylabel=None, fontsize=None, **kwargs):
        """
        Create a box plot.
        
        Args:
            Similar to plot_bar
            
        Returns:
            The matplotlib axis with the plot
        """
        fontsize = fontsize or self.FONTSIZE
        
        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(self.HEIGHT * self.ASPECT, self.HEIGHT))
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Set default order for x-axis if it's a model column
        if order is None and x == 'model':
            order = [m for m in self.model_order if m in df[x].unique()]
        
        # Get color palette
        if palette is None and hue is not None:
            palette = self._get_color_palette(df, hue)
            
        # Create box plot
        sns.boxplot(
            data=df, x=x, y=y, hue=hue, ax=ax,
            palette=palette, order=order,
            **kwargs
        )
        
        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize)
            
        # Apply theme
        self._set_theme(ax)
        
        # Format tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize-2)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
        
        return ax
    
    def plot_violin(self, data, x, y, hue=None, ax=None, palette=None, order=None,
                  title=None, xlabel=None, ylabel=None, fontsize=None, **kwargs):
        """
        Create a violin plot.
        
        Args:
            Similar to plot_bar
            
        Returns:
            The matplotlib axis with the plot
        """
        fontsize = fontsize or self.FONTSIZE
        
        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(self.HEIGHT * self.ASPECT, self.HEIGHT))
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Set default order for x-axis if it's a model column
        if order is None and x == 'model':
            order = [m for m in self.model_order if m in df[x].unique()]
        
        # Get color palette
        if palette is None and hue is not None:
            palette = self._get_color_palette(df, hue)
            
        # Create violin plot
        sns.violinplot(
            data=df, x=x, y=y, hue=hue, ax=ax,
            palette=palette, order=order,
            **kwargs
        )
        
        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize)
            
        # Apply theme
        self._set_theme(ax)
        
        # Format tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize-2)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
        
        return ax
    
    def plot_kde(self, data, x, hue=None, ax=None, palette=None, order=None,
                title=None, xlabel=None, ylabel=None, fontsize=None, **kwargs):
        """
        Create a KDE plot.
        
        Args:
            Similar to plot_bar
            
        Returns:
            The matplotlib axis with the plot
        """
        fontsize = fontsize or self.FONTSIZE
        
        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(self.HEIGHT * self.ASPECT, self.HEIGHT))
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Get color palette
        if palette is None and hue is not None:
            palette = self._get_color_palette(df, hue)
        
        # Set order of hue values if it's models
        if order is None and hue == 'model':
            order = [m for m in self.model_order if m in df[hue].unique()]
            
        # Create KDE plot
        sns.kdeplot(
            data=df, x=x, hue=hue, ax=ax,
            palette=palette, hue_order=order,
            **kwargs
        )
        
        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel or "Density", fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize)
            
        # Apply theme
        self._set_theme(ax)
        
        # Format tick labels
        plt.setp(ax.get_xticklabels(), fontsize=fontsize-2)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
        
        return ax
    
    def plot_scatter(self, data, x, y, hue=None, ax=None, palette=None, order=None, 
                    title=None, xlabel=None, ylabel=None, fontsize=None, 
                    add_regression=False, **kwargs):
        """
        Create a scatter plot.
        
        Args:
            Similar to plot_bar, plus:
            add_regression: Whether to add a regression line
            
        Returns:
            The matplotlib axis with the plot
        """
        fontsize = fontsize or self.FONTSIZE
        
        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(self.HEIGHT * self.ASPECT, self.HEIGHT))
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Get color palette
        if palette is None and hue is not None:
            palette = self._get_color_palette(df, hue)
        
        # Set order of hue values if it's models
        if order is None and hue == 'model':
            order = [m for m in self.model_order if m in df[hue].unique()]
            
        # Create scatter plot
        sns.scatterplot(
            data=df, x=x, y=y, hue=hue, ax=ax,
            palette=palette, hue_order=order,
            alpha=self.style_params["alpha"],
            **kwargs
        )
        
        # Add regression line if requested
        if add_regression:
            sns.regplot(
                data=df, x=x, y=y, 
                scatter=False, ax=ax, color='black', 
                line_kws={'linestyle':'--', 'linewidth':1}
            )
            
            # Calculate and display correlation
            corr = df[[x, y]].corr().iloc[0, 1]
            ax.text(
                0.05, 0.95, f'$r = {corr:.2f}$', 
                transform=ax.transAxes, ha='left', va='top',
                fontsize=fontsize-2, bbox=dict(facecolor='white', alpha=0.7)
            )
        
        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize)
            
        # Apply theme
        self._set_theme(ax)
        
        # Format tick labels
        plt.setp(ax.get_xticklabels(), fontsize=fontsize-2)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
        
        return ax
    
    def plot_line(self, data, x, y, hue=None, ax=None, palette=None, order=None,
                 title=None, xlabel=None, ylabel=None, fontsize=None, **kwargs):
        """
        Create a line plot.
        
        Args:
            Similar to plot_bar
            
        Returns:
            The matplotlib axis with the plot
        """
        fontsize = fontsize or self.FONTSIZE
        
        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(self.HEIGHT * self.ASPECT, self.HEIGHT))
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Get color palette
        if palette is None and hue is not None:
            palette = self._get_color_palette(df, hue)
        
        # Set order of hue values if it's models
        if order is None and hue == 'model':
            order = [m for m in self.model_order if m in df[hue].unique()]
            
        # Create line plot
        sns.lineplot(
            data=df, x=x, y=y, hue=hue, ax=ax,
            palette=palette, hue_order=order,
            **kwargs
        )
        
        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize)
            
        # Apply theme
        self._set_theme(ax)
        
        # Format tick labels
        plt.setp(ax.get_xticklabels(), fontsize=fontsize-2)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
        
        return ax
    
    def plot_heatmap(self, data, ax=None, cmap="YlGnBu", annot=True, 
                    title=None, xlabel=None, ylabel=None, fontsize=None, **kwargs):
        """
        Create a heatmap.
        
        Args:
            data: DataFrame or array for heatmap
            ax: Matplotlib axis to plot on
            cmap: Colormap for heatmap
            annot: Whether to annotate cells
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            fontsize: Font size for labels
            **kwargs: Additional keyword arguments for sns.heatmap
            
        Returns:
            The matplotlib axis with the plot
        """
        fontsize = fontsize or self.FONTSIZE
        
        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(self.HEIGHT * self.ASPECT, self.HEIGHT))
        
        # Create heatmap
        sns.heatmap(
            data=data, ax=ax, cmap=cmap, annot=annot,
            annot_kws={"size": fontsize-4},
            **kwargs
        )
        
        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize)
            
        # Format tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize-2)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
        
        return ax

    def plot_metric_grid(self, data, metrics, by="model", ncols=2, height=None, aspect=None, 
                        plot_type="box", output_dir=None, filename=None):
        """
        Create a grid of plots for multiple metrics.
        
        Args:
            data: DataFrame with metrics data
            metrics: List of metrics to plot
            by: Column to group by ("model" or "dataset")
            ncols: Number of columns in the grid
            height: Height of each subplot
            aspect: Aspect ratio of each subplot
            plot_type: Type of plot ("box", "violin", "bar", "kde")
            output_dir: Directory to save plot
            filename: Filename for saved plot
            
        Returns:
            matplotlib Figure
        """
        height = height or self.HEIGHT
        aspect = aspect or self.ASPECT
        
        # Filter metrics that are present in the data
        metrics = [m for m in metrics if m in data.columns]
        
        if len(metrics) == 0:
            print(f"{log_time()} Error: No valid metrics provided")
            return None
            
        # Calculate grid dimensions
        nrows = (len(metrics) + ncols - 1) // ncols
        
        # Create figure and axes
        fig, axes = self.create_figure_grid(nrows=nrows, ncols=ncols, height=height, aspect=aspect)
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Get plotting function based on plot_type
        if plot_type == "box":
            plot_func = self.plot_box
        elif plot_type == "violin":
            plot_func = self.plot_violin
        elif plot_type == "bar":
            plot_func = self.plot_bar
        elif plot_type == "kde":
            plot_func = self.plot_kde
        else:
            print(f"{log_time()} Warning: Unknown plot type '{plot_type}', using box plot")
            plot_func = self.plot_box
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            # Get nice metric name
            metric_name = self.rename_metric.get(metric, metric)
            
            # Create plot
            if by == "model" and "model" in df.columns:
                plot_func(
                    data=df, x="model", y=metric, 
                    ax=ax, 
                    title=metric_name,
                    xlabel="",
                    ylabel=metric_name,
                    order=[m for m in self.model_order if m in df["model"].unique()]
                )
            elif by == "dataset" and "dataset" in df.columns:
                plot_func(
                    data=df, x="dataset", y=metric, 
                    ax=ax, 
                    title=metric_name,
                    xlabel="",
                    ylabel=metric_name
                )
            else:
                # Can't group by the requested column
                print(f"{log_time()} Warning: Can't group by '{by}', using index")
                plot_func(
                    data=df, x=df.index, y=metric, 
                    ax=ax, 
                    title=metric_name,
                    xlabel="",
                    ylabel=metric_name
                )
        
        # Remove empty subplots
        for i in range(len(metrics), nrows * ncols):
            row = i // ncols
            col = i % ncols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                metrics_str = "-".join([m.replace('_', '-') for m in metrics[:3]])
                if len(metrics) > 3:
                    metrics_str += "-etc"
                filename = f"{plot_type}_{by}_{metrics_str}.pdf"
                
            output_file = output_dir / filename
            plt.savefig(output_file, bbox_inches='tight', dpi=self.style_params["figure_dpi"])
            print(f"{log_time()} Saved plot to {output_file}")
        
        return fig

    def plot_model_comparison(self, data, metric, by_dataset=False, plot_type="bar",
                             height=None, aspect=None, output_dir=None, filename=None):
        """
        Create a comparison plot of a metric across models.
        
        Args:
            data: DataFrame with metrics data
            metric: Metric to plot
            by_dataset: Whether to facet by dataset
            plot_type: Type of plot ("bar", "box", "violin")
            height: Plot height
            aspect: Plot aspect ratio
            output_dir: Directory to save the plot
            filename: Filename for saved plot
            
        Returns:
            matplotlib Figure or FacetGrid
        """
        height = height or self.HEIGHT
        aspect = aspect or self.ASPECT
        
        # Ensure model column is in the dataframe
        if 'model' not in data.columns:
            print(f"{log_time()} Error: 'model' column missing from dataframe")
            return None
            
        # Prepare data
        df = self._prepare_data(data)
            
        # Get plotting function based on plot_type
        if plot_type == "bar":
            plot_func = self.plot_bar
        elif plot_type == "box":
            plot_func = self.plot_box
        elif plot_type == "violin":
            plot_func = self.plot_violin
        else:
            print(f"{log_time()} Warning: Unknown plot type '{plot_type}', using bar plot")
            plot_func = self.plot_bar
        
        # Get model order
        model_order = [m for m in self.model_order if m in df['model'].unique()]
        
        # Get nice metric name
        metric_name = self.rename_metric.get(metric, metric)
        
        if by_dataset and 'dataset' in df.columns:
            # Create faceted plot by dataset
            datasets = df['dataset'].unique()
            
            if len(datasets) > 1:
                # Multiple datasets, create facet grid
                ncols = min(len(datasets), 3)
                nrows = (len(datasets) + ncols - 1) // ncols
                
                fig, axes = self.create_figure_grid(
                    nrows=nrows, ncols=ncols, 
                    height=height, aspect=aspect
                )
                
                for i, dataset in enumerate(datasets):
                    row = i // ncols
                    col = i % ncols
                    ax = axes[row, col]
                    
                    # Filter data for this dataset
                    dataset_df = df[df['dataset'] == dataset]
                    
                    # Get dataset display name
                    dataset_name = dataset_df['dataset_display'].iloc[0] if 'dataset_display' in dataset_df.columns else dataset
                    
                    # Create plot
                    plot_func(
                        data=dataset_df, 
                        x='model', y=metric, 
                        ax=ax, 
                        title=dataset_name,
                        xlabel="" if row < nrows - 1 else "Model",
                        ylabel=metric_name if col == 0 else "",
                        order=model_order
                    )
                
                # Remove empty subplots
                for i in range(len(datasets), nrows * ncols):
                    row = i // ncols
                    col = i % ncols
                    fig.delaxes(axes[row, col])
                    
                plt.tight_layout()
                    
            else:
                # Only one dataset, create simple plot
                fig, ax = plt.subplots(figsize=(height * aspect, height))
                
                dataset = datasets[0]
                dataset_df = df[df['dataset'] == dataset]
                
                # Get dataset display name
                dataset_name = dataset_df['dataset_display'].iloc[0] if 'dataset_display' in dataset_df.columns else dataset
                
                # Create plot
                plot_func(
                    data=dataset_df, 
                    x='model', y=metric, 
                    ax=ax, 
                    title=dataset_name,
                    xlabel="Model",
                    ylabel=metric_name,
                    order=model_order
                )
                
        else:
            # Simple comparison across all data
            fig, ax = plt.subplots(figsize=(height * aspect, height))
            
            # Aggregate data by model if needed
            if df.groupby('model')[metric].count().max() > 1:
                # Multiple values per model, show distribution
                if plot_type == "bar":
                    # For bar plots, use mean values
                    agg_df = df.groupby('model')[metric].mean().reset_index()
                    plot_func(
                        data=agg_df, 
                        x='model', y=metric, 
                        ax=ax, 
                        xlabel="Model",
                        ylabel=metric_name,
                        order=model_order
                    )
                else:
                    # For box/violin plots, use all data
                    plot_func(
                        data=df, 
                        x='model', y=metric, 
                        ax=ax, 
                        xlabel="Model",
                        ylabel=metric_name,
                        order=model_order
                    )
            else:
                # One value per model, use bar plot
                plot_func(
                    data=df, 
                    x='model', y=metric, 
                    ax=ax, 
                    xlabel="Model",
                    ylabel=metric_name,
                    order=model_order
                )
        
        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                dataset_str = "by-dataset" if by_dataset else "all-datasets"
                metric_str = metric.replace('_', '-')
                filename = f"{plot_type}_{metric_str}_{dataset_str}_comparison.pdf"
                
            output_file = output_dir / filename
            plt.savefig(output_file, bbox_inches='tight', dpi=self.style_params["figure_dpi"])
            print(f"{log_time()} Saved plot to {output_file}")
        
        return fig

    def plot_distribution_grid(self, data, metrics=None, by="model", plot_type="kde", 
                              height=None, aspect=None, output_dir=None, filename=None):
        """
        Create a grid of distribution plots.
        
        Args:
            data: DataFrame with metrics data
            metrics: List of metrics to plot (if None, uses all numeric columns)
            by: Column to group distributions by ("model" or "dataset")
            plot_type: Type of plot ("kde" or "violin")
            height: Height of each subplot
            aspect: Aspect ratio of each subplot
            output_dir: Directory to save plot
            filename: Filename for saved plot
            
        Returns:
            matplotlib Figure
        """
        height = height or self.HEIGHT
        aspect = aspect or self.ASPECT * 1.5  # Wider for distributions
        
        # Prepare data
        df = self._prepare_data(data)
        
        # If no metrics provided, use all numeric columns
        if metrics is None:
            metrics = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        else:
            # Filter metrics that are present in the data
            metrics = [m for m in metrics if m in df.columns]
        
        if len(metrics) == 0:
            print(f"{log_time()} Error: No valid metrics provided")
            return None
        
        if by not in df.columns:
            print(f"{log_time()} Error: Column '{by}' not found in data")
            return None
        
        # Maximum of 6 metrics for readability
        if len(metrics) > 6:
            print(f"{log_time()} Warning: Limiting to 6 metrics for readability")
            metrics = metrics[:6]
            
        # Calculate grid dimensions (2 columns)
        ncols = 2
        nrows = (len(metrics) + ncols - 1) // ncols
        
        # Create figure and axes
        fig, axes = self.create_figure_grid(nrows=nrows, ncols=ncols, height=height, aspect=aspect)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            # Get nice metric name
            metric_name = self.rename_metric.get(metric, metric)
            
            # Create plot
            if plot_type == "kde":
                self.plot_kde(
                    data=df, x=metric, hue=by,
                    ax=ax,
                    xlabel=metric_name,
                    ylabel="Density"
                )
            elif plot_type == "violin":
                self.plot_violin(
                    data=df, x=by, y=metric,
                    ax=ax,
                    xlabel="",
                    ylabel=metric_name
                )
            else:
                print(f"{log_time()} Warning: Unknown plot type '{plot_type}', using kde")
                self.plot_kde(
                    data=df, x=metric, hue=by,
                    ax=ax,
                    xlabel=metric_name,
                    ylabel="Density"
                )
        
        # Remove empty subplots
        for i in range(len(metrics), nrows * ncols):
            row = i // ncols
            col = i % ncols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                metrics_str = "-".join([m.replace('_', '-') for m in metrics[:3]])
                if len(metrics) > 3:
                    metrics_str += "-etc"
                filename = f"{plot_type}_{by}_{metrics_str}_distributions.pdf"
                
            output_file = output_dir / filename
            plt.savefig(output_file, bbox_inches='tight', dpi=self.style_params["figure_dpi"])
            print(f"{log_time()} Saved plot to {output_file}")
        
        return fig

    def plot_correlation_matrix(self, data, metrics=None, method='pearson', 
                               height=None, aspect=None, output_dir=None, filename=None):
        """
        Create a correlation matrix heatmap for metrics.
        
        Args:
            data: DataFrame with metrics data
            metrics: List of metrics to include (if None, uses all numeric columns)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            height: Plot height
            aspect: Plot aspect ratio
            output_dir: Directory to save plot
            filename: Filename for saved plot
            
        Returns:
            matplotlib Figure
        """
        height = height or self.HEIGHT * 1.5
        aspect = aspect or self.HEIGHT * 1.5
        
        # Prepare data
        df = self._prepare_data(data)
        
        # If no metrics provided, use all numeric columns
        if metrics is None:
            metrics = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        else:
            # Filter metrics that are present in the data
            metrics = [m for m in metrics if m in df.columns]
        
        if len(metrics) < 2:
            print(f"{log_time()} Error: Need at least 2 metrics for correlation matrix")
            return None
            
        # Calculate correlation matrix
        corr_df = df[metrics].corr(method=method)
        
        # Replace metric names with display names
        display_names = [self.rename_metric.get(m, m) for m in corr_df.columns]
        corr_df.columns = display_names
        corr_df.index = display_names
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(height * aspect, height))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_df, dtype=bool))  # Mask upper triangle
        self.plot_heatmap(
            data=corr_df,
            ax=ax,
            mask=mask,
            cmap="coolwarm",
            vmin=-1, vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title(f"{method.capitalize()} Correlation Matrix", fontsize=self.FONTSIZE)
        plt.tight_layout()
        
        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                filename = f"correlation_matrix_{method}.pdf"
                
            output_file = output_dir / filename
            plt.savefig(output_file, bbox_inches='tight', dpi=self.style_params["figure_dpi"])
            print(f"{log_time()} Saved plot to {output_file}")
        
        return fig

    def plot_correlation_grid(self, data, x_metrics=None, y_metrics=None, hue="model", 
                             height=None, aspect=None, output_dir=None, filename=None):
        """
        Create a grid of scatter plots showing correlations between pairs of metrics.
        
        Args:
            data: DataFrame with metrics data
            x_metrics: List of metrics for x-axes (if None, uses default metrics)
            y_metrics: List of metrics for y-axes (if None, uses same as x_metrics)
            hue: Column for color-coding points (typically "model")
            height: Height of each subplot
            aspect: Aspect ratio of each subplot
            output_dir: Directory to save plot
            filename: Filename for saved plot
            
        Returns:
            matplotlib Figure
        """
        height = height or self.HEIGHT
        aspect = aspect or self.ASPECT
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Default metrics if none provided
        default_metrics = ['mean_surprisal', 'uid_variance', 'uid_pairwise']
        
        # If no x_metrics provided, use defaults
        if x_metrics is None:
            x_metrics = [m for m in default_metrics if m in df.columns]
        else:
            # Filter metrics that are present in the data
            x_metrics = [m for m in x_metrics if m in df.columns]
        
        # If no y_metrics provided, use same as x_metrics
        if y_metrics is None:
            y_metrics = x_metrics
        else:
            # Filter metrics that are present in the data
            y_metrics = [m for m in y_metrics if m in df.columns]
        
        if len(x_metrics) == 0 or len(y_metrics) == 0:
            print(f"{log_time()} Error: No valid metrics provided")
            return None
            
        # Create figure and axes
        fig, axes = self.create_figure_grid(
            nrows=len(y_metrics), ncols=len(x_metrics), 
            height=height, aspect=aspect
        )
        
        # Plot each metric pair
        for i, y_metric in enumerate(y_metrics):
            for j, x_metric in enumerate(x_metrics):
                ax = axes[i, j]
                
                # Skip if same metric (diagonal)
                if x_metric == y_metric:
                    # Show histogram/KDE on diagonal
                    self.plot_kde(
                        data=df, x=x_metric,
                        ax=ax,
                        xlabel="" if i < len(y_metrics) - 1 else self.rename_metric.get(x_metric, x_metric),
                        ylabel=self.rename_metric.get(y_metric, y_metric) if j == 0 else ""
                    )
                else:
                    # Show scatter plot with regression line
                    self.plot_scatter(
                        data=df, x=x_metric, y=y_metric, hue=hue,
                        ax=ax,
                        xlabel="" if i < len(y_metrics) - 1 else self.rename_metric.get(x_metric, x_metric),
                        ylabel=self.rename_metric.get(y_metric, y_metric) if j == 0 else "",
                        add_regression=True
                    )
        
        plt.tight_layout()
        
        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                x_str = "-".join([m.replace('_', '-') for m in x_metrics[:2]])
                if len(x_metrics) > 2:
                    x_str += "-etc"
                    
                y_str = "-".join([m.replace('_', '-') for m in y_metrics[:2]])
                if len(y_metrics) > 2:
                    y_str += "-etc"
                    
                filename = f"correlation_grid_{x_str}_vs_{y_str}.pdf"
                
            output_file = output_dir / filename
            plt.savefig(output_file, bbox_inches='tight', dpi=self.style_params["figure_dpi"])
            print(f"{log_time()} Saved plot to {output_file}")
        
        return fig

    def create_report(self, data, output_dir, filename="uid_report.pdf"):
        """
        Create a comprehensive report with multiple plots.
        
        Args:
            data: DataFrame with metrics data
            output_dir: Directory to save the report
            filename: Filename for the report
            
        Returns:
            List of generated figures
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Identify available metrics
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        uid_metrics = [m for m in ['uid_variance', 'uid_pairwise', 'mean_surprisal'] if m in df.columns]
        
        figures = []
        
        # 1. Model comparison plots for each UID metric
        for metric in uid_metrics:
            fig = self.plot_model_comparison(
                data=df, metric=metric, by_dataset=True, plot_type="box",
                output_dir=output_dir, filename=f"1_model_comparison_{metric.replace('_', '-')}.pdf"
            )
            figures.append(fig)
        
        # 2. Distribution plots for each UID metric
        fig = self.plot_distribution_grid(
            data=df, metrics=uid_metrics, by="model", plot_type="kde",
            output_dir=output_dir, filename="2_distribution_grid.pdf"
        )
        figures.append(fig)
        
        # 3. Correlation plots
        if len(uid_metrics) >= 2:
            fig = self.plot_correlation_grid(
                data=df, x_metrics=uid_metrics, y_metrics=uid_metrics, hue="model",
                output_dir=output_dir, filename="3_correlation_grid.pdf"
            )
            figures.append(fig)
            
            fig = self.plot_correlation_matrix(
                data=df, metrics=uid_metrics + ['tokens', 'vocab_size'],
                output_dir=output_dir, filename="4_correlation_matrix.pdf"
            )
            figures.append(fig)
        
        # 4. Additional metrics plots
        additional_metrics = [m for m in ['tokens', 'vocab_size', 'sentence_length'] if m in df.columns]
        if additional_metrics:
            fig = self.plot_metric_grid(
                data=df, metrics=additional_metrics, by="model", plot_type="box",
                output_dir=output_dir, filename="5_additional_metrics.pdf"
            )
            figures.append(fig)
        
        print(f"{log_time()} Generated {len(figures)} plots for the report in {output_dir}")
        return figures