(function($){
    $(function(){
        $('.button-collapse').sideNav();

        // Ajax Form Submit functionality
        var targetForm = '#prediction-form';
        var targetId = '#prediction-results';
        var targetDiv = $(targetId);
        var targetUrl = $(targetForm).attr('action');

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
    }); // end of document ready
})(jQuery); // end of jQuery name space
