const image_input1 = document.querySelector("#image_input1")
var uploaded_image1 = "";
let output_path = "../static/assests/output.jpg"
let output = document.getElementById("output")
let rgbhistogram = document.getElementById("rgbpath")
let rgbhistogram_path = "../static/assests/rgbhistogram.jpg"
let rgbhistogram1 = document.getElementById("rgbpath1")
let rgbhistogram_path1 = "../static/assests/rgbhistogram1.jpg"
let rgbhistogram2 = document.getElementById("rgbpath2")
let rgbhistogram_path2 = "../static/assests/rgbhistogram2.jpg"
let histogram = document.getElementById("histo")
let histogram_path = "../static/assests/histogram.jpg"
let cum_dist = document.getElementById("cum_dist")
let cum_dist_path = "../static/assests/cum_dist.jpg"


image_input1.addEventListener("change", function(e) {
    const reader = new FileReader()

    reader.addEventListener("load", () => {
        uploaded_image1 = reader.result;
        document.querySelector("#display_image1").style.backgroundImage = `url(${uploaded_image1})`
        

  })


    reader.readAsDataURL(this.files[0])
})

let is_uploaded1 = false

$('#image_input1').change(function () {
    uploadImage('#upload-image1-form')
    is_uploaded1 = true 
    update_element(rgbhistogram,rgbhistogram_path,is_uploaded1)
    update_element(rgbhistogram1,rgbhistogram_path1,is_uploaded1)
    update_element(rgbhistogram2,rgbhistogram_path2,is_uploaded1)
    
    update_element(histogram,histogram_path,is_uploaded1)
    update_element(cum_dist,cum_dist_path,is_uploaded1)
    update_element(document.getElementById("normalizationimg"),'..//static//assests//normalize.jpg',is_uploaded1)
    update_element(document.getElementById("equalizationimg"),'..//static//assests//equalize.jpg',is_uploaded1)
    update_element(document.getElementById("global"),'..//static//assests//global.jpg',is_uploaded1)
    update_element(document.getElementById("local"),'..//static//assests//local.jpg',is_uploaded1)

    applyfilter()

});

$('#noiseType').change(function () {  
    applyfilter()



});
$('#smoothingType').change(function () {
    applyfilter()



});
$('#edgeType').change(function () {
    applyfilter()



});

$('#freq').change(function () {
    applyfilter()

});



//function that takes the element and a url, and updates it 
let applyfilter = () => {
    data = [is_uploaded1 , document.getElementById("noiseType").value, document.getElementById("smoothingType").value, document.getElementById("edgeType").value,document.getElementById("freq").value];
    imgProcessing(data)
    update_element(output, output_path,is_uploaded1)



}

let uploadImage = (formElement) => {
    let form_data = new FormData($(formElement)[0]);
        $.ajax({
            type: 'POST',
            url: 'http://127.0.0.1:5000//uploadImage',
            data: form_data,
            cache: false,
            contentType: false,
            processData: false,
            async: false,
            success: function(data) {
 
                console.log('Success!');
            },
        });
    
}

let uploadHybrid= (formElement) => {
    let form_data = new FormData($(formElement)[0]);
        $.ajax({
            type: 'POST',
            url: 'http://127.0.0.1:5000//uploadHybrid',
            data: form_data,
            cache: false,
            contentType: false,
            processData: false,
            async: false,
            success: function(data) {
 
                console.log('Success!');
            },
        });
    
}


let imgProcessing = (formElement) => {


            $.ajax({
                type: 'POST',
                url: 'http://127.0.0.1:5000//imgProcessing',
                data: JSON.stringify({formElement}),
                cache: false,
                dataType: 'json',
                async: false,
                contentType: 'application/json',
                processData: false,
                success: function(data) {
                    console.log(data
                        );

                  
                },
        });
    
}

//function that takes the element and a url, and updates it 
let update_element = (imgElement, imgURL,is_uploaded1) => {
    // create a new timestamp 
    setTimeout(() => {
        let timestamp = new Date().getTime();
        let queryString = "?t=" + timestamp;
        if(is_uploaded1){
            imgElement.style.backgroundImage = "url(" + imgURL + queryString + ")"};
    
    }, 800)
}


hybrid_img1.addEventListener("change", function(e) {
    const reader = new FileReader()

    reader.addEventListener("load", () => {
        uploaded_image1 = reader.result;
        document.querySelector("#hybrid_image1").style.backgroundImage = `url(${uploaded_image1})`
        

  })


    reader.readAsDataURL(this.files[0])
})

hybrid_img2.addEventListener("change", function(e) {
    const reader = new FileReader()

    reader.addEventListener("load", () => {
        uploaded_image1 = reader.result;
        document.querySelector("#hybrid_image2").style.backgroundImage = `url(${uploaded_image1})`
        

  })


    reader.readAsDataURL(this.files[0])
})

let hybrid1 = false
let hybrid2 = false
let h_path = "..//static//assests//hybridoutput.jpg"
let h = document.getElementById("hybrid_output")
$('#hybrid_img1').change(function () {
    uploadHybrid('#upload-image2-form')
    hybrid1 = true 
    data = [hybrid1 , hybrid2];
    hybrid(data)
    setTimeout(() => {
        let timestamp = new Date().getTime();
        let queryString = "?t=" + timestamp;
        if(   hybrid1 == true  &&     hybrid2 == true ){
            h.style.backgroundImage = "url(" + h_path + queryString + ")"};   
    }, 800)
});



$('#hybrid_img2').change(function () {
    uploadHybrid('#upload-image3-form')
    hybrid2 = true
    data = [hybrid1 , hybrid2];
    hybrid(data)
    setTimeout(() => {
        let timestamp = new Date().getTime();
        let queryString = "?t=" + timestamp;
        if(   hybrid1 == true  &&     hybrid2 == true ){
            h.style.backgroundImage = "url(" + h_path + queryString + ")"};   
    }, 800) 
    


});

let hybrid = (formElement) => {

    $.ajax({
        type: 'POST',
        url: 'http://127.0.0.1:5000//hybrid',
        data: JSON.stringify({formElement}),
        cache: false,
        dataType: 'json',
        async: false,
        contentType: 'application/json',
        processData: false,
        success: function(data) {
            console.log(data
                );   
        },
});

}
