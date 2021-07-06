
size=8
num_classes=2
batch_size=2048

w_cat=1
w_gauss=1
w_rec=1
# rec_type="mse"
rec_type="bce"
# python train_nsl.py --epochs 2000 --gaussian_size $size --num_classes $num_classes --batch_size $batch_size --learning_rate 1e-3 --w_gauss $w_gauss --w_categ $w_cat --w_rec $w_rec --rec_type $rec_type
python eval_nsl.py --epochs 2000 --gaussian_size $size --num_classes $num_classes --batch_size $batch_size --learning_rate 1e-3 --w_gauss $w_gauss --w_categ $w_cat --w_rec $w_rec --rec_type $rec_type