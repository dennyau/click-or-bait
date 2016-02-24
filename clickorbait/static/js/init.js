(function($){
    $(function(){
        $('.button-collapse').sideNav();

        // Ajax Form Submit functionality
        var targetForm = '#prediction-form';
        var targetId = '#prediction-results';
        var targetDiv = $(targetId);
        var targetUrl = $(targetForm).attr('action');

        // Pin the form
        //$('#index-banner').pushpin();

        // Takeover submit event
        $(targetForm).bind('submit', function(){
            predict();
            // Don't POST automatically
            return false;
        });

        // Ajax Call
        function predict() {
            // Predict
            $.ajax({
                method: 'POST',
                url: targetUrl,
                data: $(targetForm).serialize(),
                cache: false,
                beforeSend: function(){
                    // Trigger animation
                    Materialize.fadeInImage(targetId);
                    // Clear out prevous results
                    targetDiv.empty();
                },
                success: function(html) {
                    targetDiv.append(html);
                }
            });
        }

        // Get Boxplot
        $.ajax({
            method: 'GET',
            url: '/sentiment-boxplot/',
            success: function(html) {
                $( "#sentiment-boxplot" ).empty();
                Materialize.fadeInImage("#sentiment-boxplot");
                $( "#sentiment-boxplot" ).append( html );
            }
        });

        // Set up ScrollFire
        var go = 200;
        var options = [
            {selector: '#top-words', offset: go, callback: 'Materialize.fadeInImage("#top-words")' },
            {selector: '#model-details', offset: go, callback: 'Materialize.fadeInImage("#model-details")' },
            {selector: '#topic-details', offset: go, callback: 'Materialize.fadeInImage("#topic-details")' },
        ];
        Materialize.scrollFire(options);
    }); // end of document ready
})(jQuery); // end of jQuery name space
