import tkinter as tk
from tkinter import ttk, messagebox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ttkthemes import ThemedTk
import requests
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import numpy as np
import threading
import datetime
from scipy.optimize import minimize

# ===============================
# Konfiguracja kluczowych danych
# ===============================
CRYPTOCOMPARE_API_KEY = '501ea96bf8a344a7cfe257ba64206e13cc776dd24df469a309bb69e14fce1e5c'
TELEGRAM_BOT_TOKEN = '7681864379:AAFokmBXl_nfmEmqfm6wfaTbLqymo1GOoOg'
TELEGRAM_CHAT_ID = '2124268142'


# ===============================
# Klasa: Portfel kryptowalut
# ===============================
class Portfolio:
    def __init__(self):
        self.assets = []  # Lista aktywów w portfelu
        self.total_value = 0

    def add_asset(self, crypto_name, amount, price):
        """
        Dodaje kryptowalutę do portfela.
        """
        self.assets.append({
            'crypto_name': crypto_name,
            'amount': amount,
            'price': price
        })
        self.calculate_total_value()

    def remove_asset(self, crypto_name):
        """
        Usuwa kryptowalutę z portfela.
        """
        self.assets = [asset for asset in self.assets if asset['crypto_name'] != crypto_name]
        self.calculate_total_value()

    def calculate_total_value(self):
        """
        Oblicza całkowitą wartość portfela i aktualizuje total_value.
        """
        self.total_value = sum(asset['amount'] * asset['price'] for asset in self.assets)

    def update_prices(self):
        """
         Aktualizuje ceny kryptowalut w portfelu.
        """
        for asset in self.assets:
            asset['price'] = Portfolio.get_current_price(asset['crypto_name'], CRYPTOCOMPARE_API_KEY)

    def predict_portfolio_value(self, days):
        """
        Generuje prognozę wartości portfela na określoną liczbę dni.
        """
        total_value = self.total_value
        return list(map(lambda day: total_value * (1 + 0.01 * day), range(1, days + 1)))


    def optimize_portfolio(self):
        """
        Analizuje i sugeruje optymalizację portfela na podstawie alokacji i zmienności,
        w tym implementacja optymalizacji Markowitza
        """
        if not self.assets:
            return ["Portfel jest pusty. Nie można przeprowadzić optymalizacji."]

        historical_data = {}
        for asset in self.assets:
            dates, prices = Portfolio.get_historical_data(asset['crypto_name'], CRYPTOCOMPARE_API_KEY, limit=100)
            if not prices or len(prices) < 2:  # Sprawdzamy, czy są dane i wystarczająca liczba danych do obliczeń
                return [f"Brak wystarczających danych historycznych dla {asset['crypto_name']}."]
            historical_data[asset['crypto_name']] = prices

        if not historical_data:
            return ["Brak danych do optymalizacji."]

        # Optymalizacja Markowitza
        try:
            optimized_weights = self.markowitz_optimization(historical_data)
            if optimized_weights is not None:
                suggestions = self.generate_markowitz_suggestions(optimized_weights)
            else:
                suggestions = ["Nie można zoptymalizować portfela. Błąd podczas obliczeń."]
        except Exception as e:
             suggestions = [f"Nie można zoptymalizować portfela. Błąd: {e}"]


        return suggestions

    def markowitz_optimization(self, historical_data):
         """
         Optymalizacja portfela Markowitza.

        Args:
             historical_data (dict): Słownik z historycznymi danymi cenowymi dla każdego aktywa.

        Returns:
            np.array: Wektor optymalnych wag alokacji aktywów lub None w przypadku błędu.
        """
         prices = np.array(list(historical_data.values()))
         returns = np.diff(prices, axis=1) / prices[:, :-1] # return z cen w poszczególnych kolumnach
         mean_returns = np.array(list(map(np.mean, returns)))  # sredni zwrot aktywów
         covariance_matrix = np.cov(returns) # macierz kowariancji

         num_assets = len(self.assets)

         def portfolio_variance(weights):
             return np.dot(weights.T, np.dot(covariance_matrix, weights))

         def neg_portfolio_return(weights):
             return -np.dot(weights.T, mean_returns) # - by znaleźć max zwrot

         constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # suma wag ma sie równac 1
         bounds = tuple((0, 1) for _ in range(num_assets)) # 0 do 1 zakres wagi aktywów
         initial_weights = np.array([1/num_assets] * num_assets) # wagi startowe

         optimized_result = minimize(
             neg_portfolio_return,
             initial_weights,
             method='SLSQP',
             bounds=bounds,
             constraints=constraints
         )
         if optimized_result.success:
           return optimized_result.x
         else:
           return None

    def generate_markowitz_suggestions(self, optimized_weights):
         """
         Generuje sugestie na podstawie optymalizacji Markowitza.

         Args:
              optimized_weights (np.array): Wektor optymalnych wag alokacji aktywów.

         Returns:
              list: Lista sugestii dotyczących alokacji portfela.
         """
         suggestions = []
         for i, asset in enumerate(self.assets):
            current_weight = self.assets[i]['amount'] * self.assets[i]['price'] / self.total_value if self.total_value > 0 else 0
            optimized_weight = optimized_weights[i]
            difference = optimized_weight - current_weight
            if difference > 0.02: # zmiana conajmniej o 2%
                suggestions.append(
                    f"Zwiększ udział {asset['crypto_name']} do {optimized_weight * 100:.2f}% (z obecnych {current_weight * 100:.2f}%)."
                )
            elif difference < -0.02:
                suggestions.append(
                    f"Zmniejsz udział {asset['crypto_name']} do {optimized_weight * 100:.2f}% (z obecnych {current_weight * 100:.2f}%)."
                )
         if not suggestions:
            suggestions.append("Portfel jest zoptymalizowany zgodnie z modelem Markowitza.")
         return suggestions
    # ===============================
    # Funkcje pomocnicze
    # ===============================
    @staticmethod
    def get_crypto_list(api_key):
        """Pobiera listę kryptowalut dostępnych w API CryptoCompare."""
        url = 'https://min-api.cryptocompare.com/data/all/coinlist'
        try:
            response = requests.get(url, params={'api_key': api_key})
            response.raise_for_status()
            data = response.json()
            return sorted(coin['Symbol'] for coin in data['Data'].values())
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Błąd", f"Błąd API podczas pobierania listy kryptowalut: {e}")
            return []

    @staticmethod
    def validate_sarima_params(order, seasonal_order):
        """Sprawdza, czy parametry SARIMA są poprawne."""
        if not isinstance(order, tuple) or len(order) != 3:
            raise ValueError(f"Nieprawidłowy format 'order': {order}. Parametr musi być krotką trzech liczb.")
        if not isinstance(seasonal_order, tuple) or len(seasonal_order) != 4:
            raise ValueError(
                f"Nieprawidłowy format 'seasonal_order': {seasonal_order}. Parametr musi być krotką czterech liczb.")

        for param in order:
            if not isinstance(param, int) or param < 0:
                raise ValueError(
                    f"Nieprawidłowa wartość  {param} w 'order'. Wartości muszą być dodatnimi liczbami całkowitymi.")

        for param in seasonal_order:
            if not isinstance(param, int) or param < 0:
                raise ValueError(
                    f"Nieprawidłowa wartość {param} w 'seasonal_order'. Wartości muszą być dodatnimi liczbami całkowitymi.")

    @staticmethod
    def get_current_price(symbol, api_key):
        """Pobiera aktualną cenę kryptowaluty."""
        url = 'https://min-api.cryptocompare.com/data/price'
        params = {'fsym': symbol, 'tsyms': 'USD', 'api_key': api_key}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get('USD', 0)
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Błąd", f"Błąd API podczas pobierania ceny {symbol}: {e}")
            return 0

    @staticmethod
    def get_historical_data(symbol, api_key, limit=30):
        """Pobiera historyczne dane cenowe dla kryptowaluty."""
        url = 'https://min-api.cryptocompare.com/data/v2/histoday'
        params = {'fsym': symbol, 'tsym': 'USD', 'limit': limit, 'api_key': api_key}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json().get('Data', {}).get('Data', [])
            dates = [datetime.datetime.fromtimestamp(item['time']) for item in data]
            prices = [item['close'] for item in data]

            return dates, prices
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Błąd", f"Błąd API podczas pobierania danych historycznych {symbol}: {e}")
            return [], []

    @staticmethod
    def forecast_prices(prices, days=7):
        """
        Prognozuje ceny kryptowaluty na podstawie historycznych danych za pomocą modelu SARIMA.
        """
        try:

            Portfolio.validate_sarima_params(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            model = SARIMAX(prices, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            forecast = model_fit.get_forecast(steps=days)
            predicted_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()
            return predicted_mean, conf_int

        except ValueError as e:
            messagebox.showerror("Błąd w generowaniu  SARIMA dla", f'{e}')
            return None, None
        except Exception as e:
            print(f"Błąd prognozowania SARIMA: {e}")
            return None, None

    @staticmethod
    def send_telegram_message(message, chat_id, bot_token):
        """Wysyła wiadomość na Telegram."""
        url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
        params = {'chat_id': chat_id, 'text': message}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Błąd podczas wysyłania powiadomienia na Telegram: {e}")
            return False

    @staticmethod
    def calculate_moving_average(prices, window=7):
        """
        Oblicza średnią ruchomą.

          Args:
               prices (list): Lista wartości cenowych.
              window (int): Okno czasowe dla średniej ruchomej.

          Returns:
              list: Lista wartości średniej ruchomej.
        """
        if len(prices) < window:
            return [np.nan] * len(prices)

        moving_average = np.convolve(prices, np.ones(window), 'valid') / window
        padding = [np.nan] * (window - 1)
        return padding + list(moving_average)

    @staticmethod
    def check_price_threshold(symbol, api_key, threshold, chat_id, bot_token):
        """
         Sprawdza, czy cena kryptowaluty przekroczyła określony próg i wysyła powiadomienie na Telegram.
        """
        price = Portfolio.get_current_price(symbol, api_key)
        if price > threshold:
            message = f'Cena {symbol} przekroczyła {threshold} USD i wynosi obecnie {price:.2f} USD.'
            if Portfolio.send_telegram_message(message, chat_id, bot_token):
                print(f'Wysłano powiadomienie: {message}')
            else:
                print('Błąd podczas wysyłania powiadomienia Telegram.')
            return True
        return False


# ===============================
# Funkcje GUI
# ===============================
def create_gui(portfolio, api_key):
    root = ThemedTk(theme="arc")
    root.title("Portfel Krypto Nikusia")
    root.geometry("900x700")

    # Dodanie scrolla
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    crypto_list = Portfolio.get_crypto_list(api_key)

    # Ramka do dodawania kryptowalut
    frame_add = ttk.LabelFrame(scrollable_frame, text="Dodaj kryptowalutę do portfela", padding=10)
    frame_add.pack(fill="x", padx=10, pady=10)

    ttk.Label(frame_add, text="Wybierz kryptowalutę:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    crypto_var = tk.StringVar()
    crypto_dropdown = ttk.Combobox(frame_add, textvariable=crypto_var, state="normal")
    crypto_dropdown.grid(row=0, column=1, padx=5, pady=5)
    crypto_dropdown['values'] = crypto_list

    ttk.Label(frame_add, text="Ilość:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    amount_entry = ttk.Entry(frame_add)
    amount_entry.grid(row=1, column=1, padx=5, pady=5)

    def add_crypto():
        symbol = crypto_var.get()
        amount = amount_entry.get()

        if not symbol or not amount:
            messagebox.showerror("Błąd", "Proszę uzupełnić wszystkie pola.")
            return

        try:
            amount = float(amount)
            if amount <= 0:
                raise ValueError("Ilość musi być większa od zera.")
            price = Portfolio.get_current_price(symbol, api_key)
            if price == 0:
                raise ValueError("Nie udało się pobrać ceny.")
            portfolio.add_asset(symbol, amount, price)
            update_portfolio_table()
            messagebox.showinfo("Sukces", f"Dodano {amount} {symbol} do portfela.")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    ttk.Button(frame_add, text="Dodaj do portfela", command=add_crypto).grid(row=2, column=0, columnspan=2, pady=10)
    # Ramka portfela
    frame_portfolio = ttk.LabelFrame(scrollable_frame, text="Twój portfel", padding=10)
    frame_portfolio.pack(fill="both", expand=True, padx=10, pady=10)

    columns = ("crypto_name", "amount", "price", "value")
    portfolio_table = ttk.Treeview(frame_portfolio, columns=columns, show="headings", height=10)
    for col in columns:
        portfolio_table.heading(col, text=col.capitalize())
        portfolio_table.column(col, anchor="center")
    portfolio_table.pack(fill="both", expand=True)

    def remove_selected_crypto():
        selected_item = portfolio_table.selection()
        if not selected_item:
            messagebox.showerror("Błąd", "Proszę wybrać kryptowalutę do usunięcia.")
            return

        crypto_name = portfolio_table.item(selected_item, 'values')[0]
        portfolio.remove_asset(crypto_name)
        update_portfolio_table()
        messagebox.showinfo("Sukces", f"Usunięto {crypto_name} z portfela.")

    ttk.Button(frame_portfolio, text="Usuń wybraną kryptowalutę", command=remove_selected_crypto).pack(pady=5)

    total_value_label = ttk.Label(frame_portfolio, text="Wartość portfela: 0 USD")
    total_value_label.pack()

    def update_portfolio_table():
        for row in portfolio_table.get_children():
            portfolio_table.delete(row)
        for asset in portfolio.assets:
            portfolio_table.insert("", "end", values=(
                asset['crypto_name'],
                asset['amount'],
                asset['price'],
                asset['amount'] * asset['price']
            ))
        total_value_label.config(text=f"Wartość portfela: {portfolio.total_value:.2f} USD")

    # Ramka wykresu
    frame_chart_container = ttk.Frame(scrollable_frame)  # kontrola na widget do umiescenia calego grafiki wykresu
    frame_chart_container.pack(fill="both", expand=True, padx=10, pady=10)

    frame_chart = ttk.Frame(frame_chart_container)
    frame_chart.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def update_chart():
        symbol = crypto_var.get()
        if not symbol:
            messagebox.showerror("Błąd", "Proszę wybrać kryptowalutę.")
            return

        # Definicja globalnego selektora, aby był widoczny w calej funkcji update_chart
        global span_selector

        def fetch_data_and_update_chart():
            dates, prices = Portfolio.get_historical_data(symbol, api_key, limit=30)

            if not prices:
                messagebox.showerror("Błąd", "Nie udało się pobrać danych historycznych.")
                return
            if not dates:
                messagebox.showerror("Błąd", "Nie udało się pobrać dat historycznych.")
                return

            figure = Figure(figsize=(8, 6), dpi=100)
            ax = figure.add_subplot(111)

            ax.plot(dates, prices, label="Cena", color="skyblue")

            moving_average = Portfolio.calculate_moving_average(prices)
            ax.plot(dates, moving_average, label=f"SMA 7", color="darkred", linestyle='--')

            predicted_mean, conf_int = Portfolio.forecast_prices(prices, days=7)

            if predicted_mean is not None and conf_int is not None:
                forecast_days = [dates[-1] + datetime.timedelta(days=i + 1) for i in range(len(predicted_mean))]
                ax.plot(forecast_days, predicted_mean, label="Prognoza", linestyle="--", color='grey')

                # wyliczne przedziały ufność / wygenerowne zakres z lini prognozowania - i renderuje je pod grafe
                ax.fill_between(
                    forecast_days,
                    conf_int[:, 0],
                    conf_int[:, 1],
                    color="cornflowerblue",  # Zmiana koloru na intensywny niebieski
                    alpha=0.4,  # delikatnie mniejsza transparentność
                    label="Przedział ufności",
                )

            # ustawinia wykres / opis  grafu - i opisanych  elementów osi x/y

            ax.set_title(f"Cena {symbol}")
            ax.set_xlabel("Data")
            ax.set_ylabel("Cena [USD]")

            ax.legend()
            ax.grid(True)
            figure.autofmt_xdate()
            # zerowanie - widok - w widocznego w `Frame` (nie wywoal z Frame container tylko wykresy)
            for widget in frame_chart.winfo_children():
                widget.destroy()

            canvas_plot = FigureCanvasTkAgg(figure, frame_chart)
            canvas_plot.get_tk_widget().pack(fill="both", expand=True)
            canvas_plot.draw()

            # zakres generowania , - dynamiczne canvas do danych z y osi po  na canvas widok i selekcja po myszce
            def onselect(xmin, xmax):
                indmin, indmax = np.searchsorted(dates, (xmin, xmax))
                indmax = min(len(dates) - 1, indmax)

                # Generowanie wykresu po zdefiniowanym zakresem
                thisax = figure.add_subplot(111)
                # Generowanie wybranego podzbioru danych z tablicy
                thisax.plot(dates[indmin:indmax], prices[indmin:indmax], label='Cena', color='skyblue')

                thisax.set_title(f'Cena {symbol} ({xmin.strftime("%Y-%m-%d")} - {xmax.strftime("%Y-%m-%d")})')
                thisax.set_xlabel('Data')
                thisax.set_ylabel('Cena [USD]')
                # Wyświetlaj zmienione widoki na grafikę w obrębie ""
                thisax.legend()
                thisax.grid(True)
                canvas_plot.draw()

            span_selector = SpanSelector(
                ax, onselect, "horizontal", useblit=True)

        thread = threading.Thread(target=fetch_data_and_update_chart)
        thread.start()

    frame_buttons = ttk.Frame(
        frame_chart_container)  # umieszcze przyciski / do grafu poniżej wyliczonej ramki / kontenerem .
    frame_buttons.pack(side="bottom", fill='x', padx=10, pady=10)

    ttk.Button(frame_buttons, text="Aktualizuj wykres", command=update_chart).pack(side="left", pady=10)

    def clear_chart():  # by zerowało cała  strukture canvas
        for widget in frame_chart.winfo_children():
            widget.destroy()

    ttk.Button(frame_buttons, text="Wyczyść wykres", command=clear_chart).pack(side="right",
                                                                               pady=10)  # buton widoczny i działa  odpowiednio po stronie   na ramce canvas , ma widok ale steruje czym innym i widok to canvas. (button na dole pod grafika)
    # Ramka prognoz wartości portfela
    frame_predictions = ttk.LabelFrame(scrollable_frame, text="Prognoza wartości portfela", padding=10)
    frame_predictions.pack(fill="both", expand=True, padx=10, pady=10)

    predictions_table = ttk.Treeview(frame_predictions, columns=("days", "value"), show="headings", height=3)
    predictions_table.heading("days", text="Dni")
    predictions_table.heading("value", text="Wartość (USD)")
    predictions_table.column("days", anchor="center")
    predictions_table.column("value", anchor="center")
    predictions_table.pack(fill="both", expand=True)

    def update_predictions_table():
        for row in predictions_table.get_children():
            predictions_table.delete(row)

        def fetch_predictions():
            predicted_values = portfolio.predict_portfolio_value(
                days=30)  # liniowe przewidywanie dla 3 liczb dni w kodzie do tabeli ui , wiec do niej beda zgenerowane z góry,
            # poprawne działanie i  dostarczania widoku do
            if predicted_values:
                for i, days in enumerate([1, 7,
                                          30]):  # zakres for ( bo tutaj sa problemy wyliczeń  z typów ` list`, - a ma byc do 3 el) / bo  kod na w  fetchPredictions ma poprawny i kontrolowany przez if  - by działalo, i miało konkretne wyliczenie na element.  Z listy "wyciętej" do elementu.
                    if len(predicted_values) > i:  # Dodane  dopasowanie elementu
                        predictions_table.insert("", "end", values=(days, f"{predicted_values[i]:.2f}"))

        thread = threading.Thread(target=fetch_predictions)
        thread.start()

    ttk.Button(frame_predictions, text="Aktualizuj prognozy", command=update_predictions_table).pack(pady=10)
    # Ramka optymalizacji portfela
    frame_optimization = ttk.LabelFrame(scrollable_frame, text="Optymalizacja portfela", padding=10)
    frame_optimization.pack(fill="both", expand=True, padx=10, pady=10)

    optimization_text = tk.Text(frame_optimization, height=5, wrap="word")
    optimization_text.pack(fill="both", expand=True)

    def update_optimization():
        def fetch_suggestions():
            suggestions = portfolio.optimize_portfolio()
            optimization_text.delete("1.0", tk.END)
            if suggestions:
                optimization_text.insert(tk.END, "\n".join(suggestions))
            else:
                optimization_text.insert(tk.END, "Portfel jest zoptymalizowany.")

        thread = threading.Thread(target=fetch_suggestions)
        thread.start()

    ttk.Button(frame_optimization, text="Zaktualizuj optymalizację", command=update_optimization).pack(pady=10)

    # Ramka powiadomień
    frame_notifications = ttk.LabelFrame(scrollable_frame, text="Powiadomienia o cenach", padding=10)
    frame_notifications.pack(fill="both", expand=True, padx=10, pady=10)

    ttk.Label(frame_notifications, text="Wybierz kryptowalutę:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    notify_crypto_var = tk.StringVar()
    notify_crypto_dropdown = ttk.Combobox(frame_notifications, textvariable=notify_crypto_var, state="normal")
    notify_crypto_dropdown.grid(row=0, column=1, padx=5, pady=5)
    notify_crypto_dropdown['values'] = crypto_list

    ttk.Label(frame_notifications, text="Próg cenowy (USD):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    price_threshold_entry = ttk.Entry(frame_notifications)
    price_threshold_entry.grid(row=1, column=1, padx=5, pady=5)

    def refresh_portfolio_value():

        portfolio.update_prices()
        update_portfolio_table()
        root.after(10000, refresh_portfolio_value)

    def set_notification():
        symbol = notify_crypto_var.get()
        threshold = price_threshold_entry.get()

        if not symbol:
            messagebox.showerror("Błąd", "Proszę wybrać kryptowalutę.")
            return
        if not threshold:
            messagebox.showerror("Błąd", "Proszę podać próg cenowy.")
            return

        try:
            threshold = float(threshold)
            if threshold <= 0:
                messagebox.showerror("Błąd", "Próg cenowy musi być dodatni.")
                return
            if Portfolio.check_price_threshold(symbol, api_key, threshold, TELEGRAM_CHAT_ID, TELEGRAM_BOT_TOKEN):
                messagebox.showinfo("Sukces", f"Powiadomienie ustawione dla {symbol} powyżej {threshold} USD.")
            else:
                messagebox.showinfo("Informacja",f"Powiadomienie zostało ustawione, ale cena {symbol} nie przekroczyła {threshold} USD.")
        except ValueError:
            messagebox.showerror("Błąd", "Nieprawidłowy format progu cenowego. Wprowadź liczbę.")

    ttk.Button(frame_notifications, text="Ustaw powiadomienie", command=set_notification).grid(row=2, column=0, columnspan=2, pady=10)
    # Inicjalizacja danych
    update_portfolio_table()
    update_predictions_table()
    update_optimization()
    refresh_portfolio_value()

    root.mainloop()


# ===============================
# Główna funkcja aplikacji
# ===============================
def main():
    """Uruchamia aplikację Crypto Portfolio Manager."""
    portfolio = Portfolio()
    create_gui(portfolio, CRYPTOCOMPARE_API_KEY)


if __name__ == "__main__":
    main()