/* -*- mode:C; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*- */

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"


void save_detections(image im, char *fname, int num, float thresh, box *boxes, float **probs, char **names, int classes)
{
    int i;

    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){

            int width = im.h * .012;

            if(0){
                width = pow(prob, 1./2.)*10+1;
            }

            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            printf("{\"fname\": \"%s\", \"name\": \"%s\", \"conf\": %.0f, \"box\": [%d %d %d %d]},\n", 
                   fname,
                   names[class],
                   prob*100,
                   left, top, right, bot);
        }
    }
}

void run_segmentor(char *datacfg, char *cfgfile, char *weightfile, char *filename, 
                   char *output_filename, float thresh, float hier_thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    char obuff[256];
    char *output = obuff;
    int j;
    float nms=.4;
    int idx = 0;
    int has_percent = strchr(filename, '%');
    while(1){
        if (has_percent) {
            int p;
            for (p=0; p<10; p++) {
                snprintf(input, 256, filename, idx);
                struct stat bluff;
                if (stat(input, &bluff) == 0) {
                    snprintf(output, 256, output_filename, idx);
                    break;
                }
                idx++;
            }
            if (p==10) {
                return;
            }
        } else {
            strncpy(input, filename, 256);
            strncpy(output, output_filename, 256);
        }
        idx++;
        printf("Working on %s\n", input);
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
        if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

        save_detections(im, input, l.w*l.h*l.n, thresh, boxes, probs, names, l.classes);

        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
        save_image(im, output);

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        if (!has_percent) break;
    }
}
