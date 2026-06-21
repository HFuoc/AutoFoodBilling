@echo off
echo =========================================
echo TRAINING FOOD MODEL (10 Epochs)
echo =========================================
python ml\training\train_cnn.py --epochs 10 --batch-size 64 --num-workers 4 --output ml\models\cnn\model_food_v2.h5

echo =========================================
echo TRAINING CELL MODEL (10 Epochs)
echo =========================================
python ml\training\bootstrap_tray_cell_models.py --skip-detector --cnn-epochs 10 --cnn-batch-size 64 --cnn-output ml\models\cnn\model_cell_v2.h5

echo =========================================
echo TRAINING COMPLETE!
echo =========================================
pause
