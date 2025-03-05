-- version = 4
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Таблица пользователей
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_name TEXT NOT NULL,
    bool_flag1 BOOLEAN DEFAULT FALSE,
    bool_flag2 BOOLEAN DEFAULT FALSE,
    bool_flag3 BOOLEAN DEFAULT FALSE
);

-- Таблица лабораторий
CREATE TABLE laboratories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lab_name TEXT NOT NULL
);

-- Таблица кернов
CREATE TABLE kerns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kern_code TEXT NOT NULL
);

-- Таблица повреждений
CREATE TABLE damages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    damage_type TEXT NOT NULL
);

-- Таблица комментариев
CREATE TABLE comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kern_id UUID REFERENCES kerns(id) ON DELETE CASCADE,
    lab_id UUID REFERENCES laboratories(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    comment_text TEXT NOT NULL,
    insert_date TIMESTAMP DEFAULT NOW()
);

-- Таблица данных кернов
CREATE TABLE kern_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    id_party UUID NOT NULL DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    insert_date TIMESTAMP DEFAULT NOW(),
    lab_id UUID REFERENCES laboratories(id) ON DELETE CASCADE,
    kern_id UUID REFERENCES kerns(id) ON DELETE CASCADE,
    damage_id UUID REFERENCES damages(id) ON DELETE CASCADE,
    confidence_model DOUBLE PRECISION NOT NULL,
    code_model TEXT NOT NULL,
    code_algorithm TEXT NOT NULL,
    code_verify TEXT NOT NULL,
    input_type TEXT NOT NULL,
    download_date TIMESTAMP DEFAULT NOW() NOT NULL,
    validation_date TIMESTAMP NOT NULL
);

--- Вставка пользователей (10 пользователей)
INSERT INTO users (user_name) 
VALUES 
    ('user_1'), ('user_2'), ('user_3'), ('user_4'), ('user_5'),
    ('user_6'), ('user_7'), ('user_8'), ('user_9'), ('user_10') 
RETURNING id;

-- Вставка лабораторий (6 лабораторий)
INSERT INTO laboratories (lab_name) 
VALUES 
    ('Lab_A'), ('Lab_B'), ('Lab_C'), ('Lab_D'), ('Lab_E'), ('Lab_F') 
RETURNING id;

-- Вставка кернов (10 образцов керна)
INSERT INTO kerns (kern_code) 
VALUES 
    ('KERN001'), ('KERN002'), ('KERN003'), ('KERN004'), ('KERN005'),
    ('KERN006'), ('KERN007'), ('KERN008'), ('KERN009'), ('KERN010') 
RETURNING id;

-- Вставка повреждений (6 типов повреждений)
INSERT INTO damages (damage_type) 
VALUES 
    ('Трещина'), ('Скол'), ('Износ'), ('Деформация'), ('Разрыв'), ('Царапина') 
RETURNING id;

-- Вставка комментариев (уникальный комментарий для каждого образца) TODO выполнить несколько раз этот блок
INSERT INTO comments (kern_id, lab_id, user_id, comment_text, insert_date) 
SELECT 
    k.id AS kern_id,
    (SELECT id FROM laboratories ORDER BY RANDOM() LIMIT 1) AS lab_id,
    (SELECT id FROM users ORDER BY RANDOM() LIMIT 1) AS user_id,
    CASE 
        WHEN ROW_NUMBER() OVER () % 5 = 0 THEN 'Требуется дополнительное исследование'
        WHEN ROW_NUMBER() OVER () % 5 = 1 THEN 'Образец в хорошем состоянии'
        WHEN ROW_NUMBER() OVER () % 5 = 2 THEN 'Обнаружены незначительные повреждения'
        WHEN ROW_NUMBER() OVER () % 5 = 3 THEN 'Рекомендуется повторный анализ'
        ELSE 'Дефекты отсутствуют'
    END AS comment_text,
    NOW() - (random() * INTERVAL '365 days') AS insert_date
FROM kerns k;

-- Вставка данных в kern_data с логически связанными датами  TODO выполнить несколько раз
INSERT INTO kern_data (
    user_id, insert_date, lab_id, kern_id, damage_id, confidence_model, 
    code_model, code_algorithm, code_verify, input_type, download_date, validation_date
) 
SELECT 
    (SELECT id FROM users ORDER BY RANDOM() LIMIT 1),
    NOW() - (random() * INTERVAL '365 days'),  -- Генерация insert_date
    (SELECT id FROM laboratories ORDER BY RANDOM() LIMIT 1),
    (SELECT id FROM kerns ORDER BY RANDOM() LIMIT 1),
    CASE 
        WHEN random() > 0.7 THEN NULL  
        ELSE (SELECT id FROM damages ORDER BY RANDOM() LIMIT 1) 
    END,
    random(),
    'Model_' || substring(md5(random()::text) from 1 for 5),
    'Algorithm_' || substring(md5(random()::text) from 1 for 5),
    CASE WHEN random() > 0.5 THEN 'Verified' ELSE 'Unverified' END,
    CASE WHEN random() > 0.5 THEN 'Manual' ELSE 'Automatic' END,
    (NOW() - (random() * INTERVAL '365 days')) + (random() * INTERVAL '5 minutes' + INTERVAL '2 minutes'),  -- Генерация download_date
    (NOW() - (random() * INTERVAL '365 days')) + (random() * INTERVAL '5 minutes' + INTERVAL '2 minutes') + (random() * INTERVAL '10 minutes' + INTERVAL '10 minutes')  -- Генерация validation_date
FROM generate_series(1, 10);

