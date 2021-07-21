# Data
MAX_TOKEN_COUNT = 256
LABEL_COLUMNS = ['824.account-management.account-access.0', '824.online-experience.updates-versions.0',
                 '824.company-brand.general-satisfaction.0', '824.company-brand.competitor.0',
                 '824.company-brand.convenience.0', '824.account-management.fingerprint-facial-recognition.0',
                 '824.staff-support.email.0', '824.attributes.cleanliness.0', '824.staff-support.agent-named.0',
                 '824.purchase-booking-experience.choice-variety.0', '824.online-experience.language.0',
                 '824.logistics-rides.speed.0', '824.attributes.size-fit.0', '824.logistics-rides.order-accuracy.0',
                 '824.attributes.taste-flavour.0']

# Training
RANDOM_SEED = 42
BERT_MODEL_NAME = 'bert-base-cased'
N_EPOCHS = 10
BATCH_SIZE = 4

# Optimizer
WARMUP_STEPS = 20
TOTAL_TRAINING_STEPS = 100

#  Classification
THRESHOLD = 0.3
