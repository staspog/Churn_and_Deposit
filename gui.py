from header import *
from model import predict_churn_and_deposit

def create_churn_and_deposit_ui(model_churn, model_deposit, scaler_churn, scaler_deposit):
    def on_predict():
        try:
            input_data_churn = [
                var_gender.get(),
                int(entry_age.get()),
                int(entry_tenure.get()),
                float(entry_balance.get()),
                int(entry_products.get()),
                int(var_credit_card.get()),
                float(entry_salary.get())
            ]

            input_data_deposit = [
                int(entry_age.get()),
                entry_marital.get(),
                entry_education.get(),
                var_default.get(),
                var_housing.get(),
                var_loan.get(),
                entry_month.get(),
                entry_day_of_week.get(),
                int(entry_duration.get()),
                int(entry_campaign.get()),
                int(entry_pdays.get()),
                int(entry_previous.get()),
                entry_poutcome.get()
            ]
            print(input_data_churn)
            print(input_data_deposit)

            
            # Преобразование в числовые значения
            input_data_deposit[1] = {'single': 0, 'married': 1, 'divorced': 2, 'unknown': 3}.get(input_data_deposit[1], -1)  # marital
            input_data_deposit[2] = {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 'professional.course': 4, 'university.degree': 5, 'illiterate': 6, 'unknown': 7}.get(input_data_deposit[2], -1)  # education
            input_data_deposit[6] = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}.get(input_data_deposit[6], -1)  # month
            input_data_deposit[7] = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}.get(input_data_deposit[7], -1)  # day_of_week
            input_data_deposit[12] = {'failure': 0, 'nonexistent': 1, 'success': 2}.get(input_data_deposit[12], -1)  # poutcome
            
            print(input_data_churn)
            print(input_data_deposit)
            
            churn_prediction, churn_probability, deposit_prediction, deposit_probability = predict_churn_and_deposit(model_churn, model_deposit, scaler_churn, scaler_deposit, input_data_churn, input_data_deposit)
            messagebox.showinfo("Результат", f"Вероятность оттока клиента: {churn_probability:.2f}%\nВероятность успешного предложения вклада: {deposit_probability:.2f}%")
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Пожалуйста, введите корректные числовые значения! Ошибка: {e}")

    root = tk.Tk()
    root.title("Bank Client Prediction - Уход клиента и Вклад")

    # Пол для churn
    tk.Label(root, text="Пол:").grid(row=0, column=0)
    var_gender = tk.IntVar()
    tk.Radiobutton(root, text="Женщина", variable=var_gender, value=0).grid(row=0, column=1)
    tk.Radiobutton(root, text="Мужчина", variable=var_gender, value=1).grid(row=0, column=2)

    tk.Label(root, text="Возраст (18-92):").grid(row=1, column=0)
    entry_age = tk.Entry(root)
    entry_age.grid(row=1, column=1)

    tk.Label(root, text="Срок в банке (>=0):").grid(row=2, column=0)
    entry_tenure = tk.Entry(root)
    entry_tenure.grid(row=2, column=1)

    tk.Label(root, text="Баланс (>=0):").grid(row=3, column=0)
    entry_balance = tk.Entry(root)
    entry_balance.grid(row=3, column=1)

    tk.Label(root, text="Количество продуктов (>=0):").grid(row=4, column=0)
    entry_products = tk.Entry(root)
    entry_products.grid(row=4, column=1)

    tk.Label(root, text="Есть ли кредитная карта (0/1):").grid(row=5, column=0)
    var_credit_card = tk.IntVar()
    tk.Radiobutton(root, text="Нет", variable=var_credit_card, value=0).grid(row=5, column=1)
    tk.Radiobutton(root, text="Да", variable=var_credit_card, value=1).grid(row=5, column=2)
    tk.Label(root, text="Озвученная зарплата:").grid(row=6, column=0)
    entry_salary = tk.Entry(root)
    entry_salary.grid(row=6, column=1)

    # Ввод данных пользователем для предсказания успешности вклада
    tk.Label(root, text="Семейное положение:").grid(row=7, column=0)
    entry_marital = ttk.Combobox(root, values=['married', 'single', 'divorced'])
    entry_marital.grid(row=7, column=1)

    tk.Label(root, text="Образование:").grid(row=8, column=0)
    entry_education = ttk.Combobox(root, values=['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree', 'illiterate'])
    entry_education.grid(row=8, column=1)

    tk.Label(root, text="Есть ли просроченный кредит (0/1):").grid(row=9, column=0)
    var_default = tk.IntVar()
    entry_default = tk.Checkbutton(root, variable=var_default)
    entry_default.grid(row=9, column=1)

    tk.Label(root, text="Есть ли жилищный кредит (0/1):").grid(row=10, column=0)
    var_housing = tk.IntVar()
    entry_housing = tk.Checkbutton(root, variable=var_housing)
    entry_housing.grid(row=10, column=1)

    tk.Label(root, text="Есть ли личный кредит (0/1):").grid(row=11, column=0)
    var_loan = tk.IntVar()
    entry_loan = tk.Checkbutton(root, variable=var_loan)
    entry_loan.grid(row=11, column=1)

    tk.Label(root, text="Месяц последнего контакта:").grid(row=12, column=0)
    entry_month = ttk.Combobox(root, values=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    entry_month.grid(row=12, column=1)

    tk.Label(root, text="День недели последнего контакта:").grid(row=13, column=0)
    entry_day_of_week = ttk.Combobox(root, values=['mon', 'tue', 'wed', 'thu', 'fri'])
    entry_day_of_week.grid(row=13, column=1)

    tk.Label(root, text="Продолжительность последнего контакта (в секундах):").grid(row=14, column=0)
    entry_duration = tk.Entry(root)
    entry_duration.grid(row=14, column=1)

    tk.Label(root, text="Количество контактов во время текущей кампании:").grid(row=15, column=0)
    entry_campaign = tk.Entry(root)
    entry_campaign.grid(row=15, column=1)

    tk.Label(root, text="Количество дней, прошедших с последнего контакта:").grid(row=16, column=0)
    entry_pdays = tk.Entry(root)
    entry_pdays.grid(row=16, column=1)

    tk.Label(root, text="Количество контактов до текущей кампании:").grid(row=17, column=0)
    entry_previous = tk.Entry(root)
    entry_previous.grid(row=17, column=1)

    tk.Label(root, text="Исход предыдущей маркетинговой кампании:").grid(row=18, column=0)
    entry_poutcome = ttk.Combobox(root, values=['failure', 'nonexistent', 'success'])
    entry_poutcome.grid(row=18, column=1)

    # Кнопка для запуска предсказаний
    tk.Button(root, text="Оценить", command=on_predict).grid(row=19, columnspan=2)
    root.mainloop()