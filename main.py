import csv
from collections import defaultdict
import matplotlib.pyplot as plt


# 연령대를 구분하기 위한 기준 정의
AGE_GROUPS = [
    ('10s', 10, 19),
    ('20s', 20, 29),
    ('30s', 30, 39),
    ('40s', 40, 49),
    ('50s', 50, 59),
    ('60s', 60, 69),
    ('70s', 70, 79),
]


# -------------------------------
# CSV 파일 읽기
# -------------------------------
def read_csv_file(file_name):
    # CSV 파일을 읽어서 각 행을 딕셔너리 형태로 저장
    rows = []

    with open(file_name, 'r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


# -------------------------------
# train 데이터와 test 데이터 병합
# -------------------------------
def merge_data(train_rows, test_rows):

    # 두 데이터를 하나의 리스트로 합치는 과정
    merged_rows = []

    # 먼저 train 데이터를 추가
    for row in train_rows:
        merged_rows.append(row)

    # test 데이터 추가
    # test 데이터에는 Transported 항목이 없기 때문에 빈 값으로 추가
    for row in test_rows:

        new_row = dict(row)

        if 'Transported' not in new_row:
            new_row['Transported'] = ''

        merged_rows.append(new_row)

    return merged_rows


# -------------------------------
# 전체 데이터 개수 확인
# -------------------------------
def print_data_count(train_rows, test_rows, merged_rows):

    print('Train data count:', len(train_rows))
    print('Test data count:', len(test_rows))
    print('Merged total data count:', len(merged_rows))


# -------------------------------
# 문자열 형태의 True / False 값을
# 실제 boolean 값으로 변환
# -------------------------------
def parse_bool(value):

    if value == 'True':
        return True

    if value == 'False':
        return False

    return None


# -------------------------------
# Transported와 가장 관련성이 높은 항목 찾기
# -------------------------------
def find_related_feature(train_rows):

    # 비교할 주요 범주형 변수들
    columns = [
        'HomePlanet',
        'CryoSleep',
        'Destination',
        'VIP'
    ]

    best_column = None
    best_score = 0

    # 각 컬럼마다 Transported 비율 차이를 계산
    for column in columns:

        counts = defaultdict(lambda: {'true': 0, 'false': 0})

        for row in train_rows:

            transported = parse_bool(row.get('Transported'))
            value = row.get(column)

            # 값이 없는 경우 제외
            if transported is None or value == '':
                continue

            if transported:
                counts[value]['true'] += 1
            else:
                counts[value]['false'] += 1

        rates = []

        # 각 값에 대한 Transported 비율 계산
        for value in counts:

            total = counts[value]['true'] + counts[value]['false']

            if total == 0:
                continue

            rate = counts[value]['true'] / total
            rates.append(rate)

        if len(rates) < 2:
            continue

        # 가장 높은 비율과 낮은 비율 차이를 점수로 사용
        score = max(rates) - min(rates)

        if score > best_score:
            best_score = score
            best_column = column

    print('\nMost related feature with Transported:', best_column)


# -------------------------------
# 나이를 연령대 그룹으로 변환
# -------------------------------
def get_age_group(age):

    if age == '' or age is None:
        return None

    try:
        age = float(age)
    except ValueError:
        return None

    # 정의된 연령대 범위에 따라 그룹 분류
    for label, start, end in AGE_GROUPS:

        if start <= age <= end:
            return label

    return None


# -------------------------------
# 연령대별 Transported 여부 시각화
# -------------------------------
def plot_age_group_transport(train_rows):

    # 각 연령대별 Transported 여부를 저장할 변수
    true_counts = {label: 0 for label, _, _ in AGE_GROUPS}
    false_counts = {label: 0 for label, _, _ in AGE_GROUPS}

    for row in train_rows:

        age_group = get_age_group(row.get('Age'))
        transported = parse_bool(row.get('Transported'))

        if age_group is None or transported is None:
            continue

        if transported:
            true_counts[age_group] += 1
        else:
            false_counts[age_group] += 1

    labels = [label for label, _, _ in AGE_GROUPS]

    true_values = [true_counts[label] for label in labels]
    false_values = [false_counts[label] for label in labels]

    x = range(len(labels))
    width = 0.35

    plt.figure()

    plt.bar([i - width / 2 for i in x], true_values, width, label='Transported True')
    plt.bar([i + width / 2 for i in x], false_values, width, label='Transported False')

    plt.xticks(x, labels)

    plt.xlabel('Age Group')
    plt.ylabel('Passenger Count')
    plt.title('Transported by Age Group')

    plt.legend()

    plt.show()


# -------------------------------
# Destination 별 연령대 분포 시각화
# -------------------------------
def plot_destination_age_distribution(rows):

    # 목적지별 연령대 분포를 저장할 딕셔너리
    data = defaultdict(lambda: {label: 0 for label, _, _ in AGE_GROUPS})

    for row in rows:

        destination = row.get('Destination')
        age_group = get_age_group(row.get('Age'))

        if destination == '' or age_group is None:
            continue

        data[destination][age_group] += 1

    destinations = list(data.keys())
    labels = [label for label, _, _ in AGE_GROUPS]

    x = range(len(labels))
    width = 0.25

    plt.figure()

    # 목적지별로 막대 그래프 생성
    for index, destination in enumerate(destinations):

        values = [data[destination][label] for label in labels]

        positions = [i + index * width for i in x]

        plt.bar(positions, values, width, label=destination)

    plt.xticks(x, labels)

    plt.xlabel('Age Group')
    plt.ylabel('Passenger Count')
    plt.title('Age Distribution by Destination')

    plt.legend()

    plt.show()


# -------------------------------
# 메인 실행 부분
# -------------------------------
def main():

    # CSV 파일 읽기
    train_rows = read_csv_file('train.csv')
    test_rows = read_csv_file('test.csv')

    # 데이터 병합
    merged_rows = merge_data(train_rows, test_rows)

    # 전체 데이터 개수 출력
    print_data_count(train_rows, test_rows, merged_rows)

    # Transported와 가장 관련성이 높은 항목 찾기
    find_related_feature(train_rows)

    # 연령대별 Transported 시각화
    plot_age_group_transport(train_rows)

    # Destination별 연령대 분포 시각화 (보너스)
    plot_destination_age_distribution(merged_rows)


if __name__ == '__main__':
    main()
