---
layout: post
title:  "Visualization: Softmax output is a quantization with dot product similarity"
date:   2021-03-04 00:00:00 +0000
categories: deep learning, visualization
---
<script src="https://d3js.org/d3.v6.min.js"></script>
<div class='chart'> </div>

<script>
function display_chart() {
            var svg = d3.select(".chart").append("svg").attr("width", 500).attr("height", 400).style("border", "1px solid black");

            

            points = []
            for (var i =-50; i < 50; i++){
                for( var k = -50; k < 50; k++){
                    points.push([k/10.0, i/10.0]);
                }
            }

            x = d3.scaleLinear()
            .domain([-5.0, 5.0])
            .range([50, 450]);

            y = d3.scaleLinear()
            .domain([-5.0, 5.0])
            .range([50, 350]);

            vecs = [[1.0, 1.0], [-0.5, 1.0]];
            
            


            var data_points = svg.selectAll('g').data(points)
            .enter()

            var vectors = svg.selectAll('g').data(vecs).enter()

            data_points.append('circle')
            .attr('cx', function(d, i) { return x(d[0]);})
            .attr('cy', function(d, i) { return y(d[1]);})
            .attr('r', 2)
            .attr()


            
            vectors.append('line')
            .attr('x1',function(d){return x(0)})
            .attr('y1',function(d){return y(0)})
            .attr('x2',function(d){return x(d[0])})
            .attr('y2',function(d){return y(d[1])})
            .attr("stroke", "green")


        }
display_chart();
</script>