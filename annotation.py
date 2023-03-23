import os
import gradio as gr
import pandas as pd
from PIL import Image


query_df = pd.DataFrame(columns=['img_path','label','labeled_yn'])
labeled_list = []
unlabeled_list = []

# select filepath
def select_file(filepath, datadir):
    global query_df
    global labeled_list
    global unlabeled_list
    
    # read a file
    query_df = pd.read_csv(filepath)
    # set unlabeled image path list
    labeled_list = query_df.loc[query_df['labeled_yn']==True, 'img_path'].tolist()
    unlabeled_list = query_df.loc[query_df['labeled_yn']==False, 'img_path'].tolist()
    
    return gr.Dropdown.update(choices=labeled_list), os.path.join(datadir, unlabeled_list[0]), nb_left_imgs()


def show_image(datadir, img_path):
    img_path, label, _ = query_df[query_df['img_path']==img_path].values[0]
    
    return f'image path: {img_path}\nlabel: {label}', Image.open(os.path.join(datadir, img_path))

def save_result(filepath, datadir, label):
    img_path = unlabeled_list[0]
    
    # save
    save_file(filepath=filepath, img_path=img_path, label=label)
    
    # remove an annotated image path from unlabeled_list 
    unlabeled_list.remove(img_path)
    
    # append an annotated image path into labeled_list
    labeled_list.append(img_path)
    
    return nb_left_imgs(), os.path.join(datadir, unlabeled_list[0]), gr.Dropdown.update(choices=labeled_list)


def save_relabel_result(filepath, datadir, label, img_path):
    # save
    save_file(filepath=filepath, img_path=img_path, label=label)

    return nb_left_imgs(), Image.open(os.path.join(datadir, unlabeled_list[0]))


def save_file(img_path, filepath, label):
    # update
    cond = query_df.img_path==img_path
    query_df.loc[cond, 'label'] = int(label)
    query_df.loc[cond, 'labeled_yn'] = True
    
    # save
    query_df.to_csv(filepath, index=False)
    
def nb_left_imgs():
    return f"{len(query_df[query_df['labeled_yn']==False])}/{len(query_df)}"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            round_path = gr.Textbox(label='Query list for annotation', value='./results/exp/MNIST/resnet18/LeastConfidence-n_query500/round1/query_list.csv')
            data_path = gr.Textbox(label='Data directory', value='./data/MNIST')
            select_btn = gr.Button('Select')
          
            left_imgs = gr.Textbox(label='Left Images')
            image_output = gr.Image(type='pil').style(height=200, width=200)
    
            # annotation
            classes = gr.Radio([str(i) for i in range(10)], label='Choice a label of the image')
            label_btn = gr.Button("Choice")
            
            # show image
            labeled_info_dropdown = gr.Dropdown(label='Image list to re-labeling')
            labeled_info = gr.Textbox(label='labeled infomation of image')
            with gr.Row():
                show_btn = gr.Button('Show')
                relabel_btn = gr.Button('Save re-label')
                
    select_btn.click(fn=select_file, inputs=[round_path, data_path], outputs=[labeled_info_dropdown, image_output, left_imgs])
    label_btn.click(fn=save_result, inputs=[round_path, data_path, classes], outputs=[left_imgs, image_output, labeled_info_dropdown])
    show_btn.click(fn=show_image, inputs=[data_path, labeled_info_dropdown], outputs=[labeled_info, image_output])
    relabel_btn.click(fn=save_relabel_result, inputs=[round_path, data_path, classes, labeled_info_dropdown], outputs=[left_imgs, image_output])
    
demo.launch(share=True)