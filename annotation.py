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
    
    # returns
    update_dropdown = gr.Dropdown.update(choices=labeled_list)
    output_img = os.path.join(datadir, unlabeled_list[0]) if len(unlabeled_list) > 0 else None
    left_txt = nb_left_imgs()
    current_img_path = unlabeled_list[0] if len(unlabeled_list) > 0 else ""
    
    return update_dropdown, output_img, left_txt, current_img_path


def show_image(datadir, img_path):
    img_path, label, _ = query_df[query_df['img_path']==img_path].values[0]
    
    return img_path, label, Image.open(os.path.join(datadir, img_path))

def save_result(filepath, datadir, label):
    
    if len(unlabeled_list) > 0:
        img_path = unlabeled_list[0]
    
        # save
        save_file(filepath=filepath, img_path=img_path, label=label)
        
        # remove an annotated image path from unlabeled_list 
        unlabeled_list.remove(img_path)
        
        # append an annotated image path into labeled_list
        labeled_list.insert(0, img_path)
    
    # returns
    left_txt = nb_left_imgs()
    output_img = os.path.join(datadir, unlabeled_list[0]) if len(unlabeled_list) > 0 else None
    update_dropdown = gr.Dropdown.update(choices=labeled_list)
    current_img_path = unlabeled_list[0] if len(unlabeled_list) > 0 else ""
    
    return left_txt, output_img, update_dropdown, current_img_path


def save_relabel_result(filepath, label, img_path):
    # save
    save_file(filepath=filepath, img_path=img_path, label=label)

    return label


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
          
            left_imgs = gr.Textbox(label='The number of unlabeled images / total images')
            
            with gr.Tab("Unlabeled"):
                # show image
                unlabeled_info = gr.Textbox(label='Current image path')
                unlabeled_image_output = gr.Image(type='pil').style(height=200, width=200)
        
                # annotation
                unlabeled_classes = gr.Radio([str(i) for i in range(10)], label='Choice a label of the image')
                label_btn = gr.Button("Choice")
            
            with gr.Tab("Labeled"):
                # show image
                labeled_info_dropdown = gr.Dropdown(label='Image list to re-labeling (sort by recent)')
                show_btn = gr.Button('Show')
                
                with gr.Row():
                    labeled_path = gr.Textbox(label='Selected image path')
                    labeled_info = gr.Textbox(label='Annotated label')
                
                # show image
                labeled_image_output = gr.Image(type='pil').style(height=200, width=200)
        
                # annotation
                labeled_classes = gr.Radio([str(i) for i in range(10)], label='Choice a label of the image')
                
                # re-labeling    
                relabel_btn = gr.Button('Save re-label')
                
    # labeling
    select_btn.click(fn=select_file, inputs=[round_path, data_path], outputs=[labeled_info_dropdown, unlabeled_image_output, left_imgs, unlabeled_info])
    label_btn.click(fn=save_result, inputs=[round_path, data_path, unlabeled_classes], outputs=[left_imgs, unlabeled_image_output, labeled_info_dropdown, unlabeled_info])

    # re-labeling
    show_btn.click(fn=show_image, inputs=[data_path, labeled_info_dropdown], outputs=[labeled_path, labeled_info, labeled_image_output])
    relabel_btn.click(fn=save_relabel_result, inputs=[round_path, labeled_classes, labeled_path], outputs=labeled_info)
    
demo.launch(share=True)