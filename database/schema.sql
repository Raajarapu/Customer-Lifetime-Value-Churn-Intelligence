
CREATE TABLE IF NOT EXISTS customers (
    customer_id BIGINT PRIMARY KEY,
    surname VARCHAR(150),
    credit_score INTEGER,
    geography VARCHAR(50),
    gender VARCHAR(20),
    age INTEGER,
    tenure INTEGER,
    balance NUMERIC(15,2),
    num_of_products INTEGER,
    has_cr_card INTEGER,
    is_active_member INTEGER,
    estimated_salary NUMERIC(15,2),
    exited INTEGER,
    complain INTEGER,
    satisfaction_score INTEGER,
    card_type VARCHAR(50),
    point_earned INTEGER
);

CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT,
    predicted_churn INTEGER,
    churn_probability NUMERIC(10,6),
    model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customer_segments (
    segment_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT,
    customer_value_score NUMERIC(10,6),
    customer_value_segment VARCHAR(50),
    cluster INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS uplift_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    treatment INTEGER,
    conversion INTEGER,
    uplift NUMERIC(12,8),
    uplift_category VARCHAR(50),
    dataset_source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value NUMERIC(15,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

