var express = require('express');
var app = express();

app.get('/', function(req, res, next) {
    // const thumb_list = ['39.png', '41.png', '65.png', '67.png', '67_2.png','73_2.png', '76.png', '73.png', '64.png']
    const thumb_list = ['73_2.png','73.png']
    res.render('thumbnail', {thumb_list});
});

app.get('/detail/:id', function(req, res, next) {
    const imageId = req.params.id;
    const detailInfo_67 = {
        label: '67.0',
        first_time: '2023-12-01 11:18:10.137639',
        last_time: '2023-12-01 11:18:10.464388',
        bbox: '[440.5381, 188.8263, 532.6275, 316.4584]',
        image: 'detail_67.png'
    };

    const detailInfo_76 = {
        label: '76.0',
        first_time: '2023-12-01 11:18:10.137639',
        last_time: '2023-12-01 11:18:10.464388',
        bbox: '[440.5381, 188.8263, 532.6275, 316.4584]',
        image: 'detail_76.png'
    };

    const detailInfo_73 = {
        label: '73.0',
        first_time: '2023-12-01 11:18:10.137639',
        last_time: '2023-12-01 11:18:10.464388',
        bbox: '[440, 188, 532, 316]',
        image: 'detail_73.png'
    };

    if (imageId == 1){
        res.render('detail', {detailInfo: detailInfo_73});
    }else if (imageId == 2){
        res.render('detail', {detailInfo: detailInfo_67});
    }
});

module.exports = app;
