# Модуль инкрементального обучения моделей нестационарных финансовых процессов. 
<p align=right><i>Divide et Impera</i></p>
Для определения моделей переходных процессов был создан модуль инкрементального обучения для идентификации моделей переходных финансовых процессов, позволяющий динамически оценивать предсказуемость последовательности транзакций пользователей банковских карт с целью преодоления снижения оценки прогнозирования на временном промежутке. Ключевыми особенностями применённого метода являются, во-первых, динамическое измерение предсказуемости поведения агентов на микроуровне с применением «измерительной» нейронной сети, а во-вторых, группировка агентов по предсказуемости поведения с целью улучшения качества результата прогнозирующей нейронной сети для всей популяции.

## Описание файлов
* sim_clients - папка с предсказаниями для всех клиентов на микроуровне, 
* Module_5.ipynb - тетрадка для проведения экспериментов,
* all_rw_n_sur.csv - траты всех клиентов,
* bad_rw_n.csv - траты плохо предсказуемых клиентов,
* good_rw_n.csv - траты хорошо предсказуемых клиентов,

* macro_model_functions.py - функции для работы с моделью на мезо- и макроуровне,
* micro_help_functions.py - вспомогательные функции для проведения эксперимента на микроуровне,
* micro_model_functions.py - функции для работы с моделью на микроуровне,
* preprocessing.py - функции для предобработки данных.
