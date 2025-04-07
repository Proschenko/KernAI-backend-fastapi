-- version = 6

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_name TEXT NOT NULL
);

-- Таблица лабораторий
CREATE TABLE public.laboratories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lab_name TEXT NOT NULL
);

INSERT INTO public.laboratories (lab_name) VALUES
    ('Лаборатория 1'),
    ('Лаборатория 2'),
    ('Лаборатория 3');

-- Таблица кернов
CREATE TABLE public.kerns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kern_code TEXT NOT NULL
);

-- Таблица повреждений
CREATE TABLE public.damages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    damage_type TEXT NOT NULL
);

INSERT INTO public.damages (damage_type) VALUES
    ('Трещина'),
    ('Скол'),
    ('Износ');

-- Таблица комментариев
CREATE TABLE public.comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kern_id UUID REFERENCES public.kerns(id) ON DELETE CASCADE,
    lab_id UUID REFERENCES public.laboratories(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    comment_text TEXT NOT NULL,
    insert_date TIMESTAMP DEFAULT now()
);

-- Таблица партий кернов
CREATE TABLE public.kern_party (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    party_id UUID NOT NULL
);

-- Таблица для хранения списка строк (многие ко многим с kern_party)
CREATE TABLE public.kern_party_statements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    party_id UUID REFERENCES public.kern_party(id) ON DELETE CASCADE,
    kern_code_from_statement TEXT NOT NULL
);

-- Таблица аналитики кернов
CREATE TABLE public.kern_data_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    confidence_model FLOAT8 NOT NULL,
    code_model TEXT NOT NULL,
    code_algorithm TEXT,
    input_type TEXT NOT NULL,
    download_date TIMESTAMP DEFAULT now() NOT NULL,
    validation_date TIMESTAMP NOT NULL
);

-- Основная таблица данных кернов
CREATE TABLE public.kern_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    id_party UUID REFERENCES public.kern_party(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    insert_date TIMESTAMP DEFAULT now(),
    lab_id UUID REFERENCES public.laboratories(id) ON DELETE CASCADE,
    kern_id UUID REFERENCES public.kerns(id) ON DELETE CASCADE,
    damage_id UUID REFERENCES public.damages(id) ON DELETE CASCADE,
    analytic_id UUID UNIQUE REFERENCES public.kern_data_analytics(id) ON DELETE CASCADE
);

