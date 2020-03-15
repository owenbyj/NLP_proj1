from pymysql import connect
from utils.utils import time_counter

conn = connect(host='localhost', user='root', password='Tues!2288', database='testdb')

cursor = conn.cursor()


def create():
    sql = """CREATE TABLE word_vector (
    WORD VARCHAR(255),"""

    for i in range(1, 300):
        sql += 'D{} DOUBLE,'.format(i)

    sql += 'D300 DOUBLE);'

    cursor.execute(sql)


@time_counter
def insert_all_word_vec():
    sql_template = 'INSERT INTO word_vector VALUES ({});'
    with open('../data/no_symbol.word2vec', 'r') as wvh:
        lines = wvh.readlines()

    index = 1
    for line in lines[1:]:
        print("INSERTING LINE {}".format(index))
        line_list = line.strip().split()
        line_list[0] = "'" + line_list[0] + "'"
        sql = sql_template.format(', '.join(line_list))
        cursor.execute(sql)
        index += 1


if __name__ == '__main__':
    insert_all_word_vec()
    conn.close()
