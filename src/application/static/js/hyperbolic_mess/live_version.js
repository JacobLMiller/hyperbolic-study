;
(function () {
  if (typeof HyperbolicCanvas === 'undefined') {
    window.HyperbolicCanvas = {};
  }
  if (typeof HyperbolicCanvas.scripts === 'undefined') {
    window.HyperbolicCanvas.scripts = {};
  }

let dragged = false;
let flag = true;
var SCROLL_SPEED = .03;
var SCALE_FACTOR = .003;
let zoom = 1;
let totalZoom = 1;

let mapNodes = [];


//counterclockwise and clockwise functions are re-used code from spherical implementation.
function toCounterClockwise(polygon) {
    var sum = 0;

    // loop through points and sum edges (x2-x1)(y2+y1)
    for (var i = 0; i + 3 < polygon.length; i += 2) {
        sum += (parseFloat(polygon[i + 2]) - parseFloat(polygon[i])) * (parseFloat(polygon[i + 3]) + parseFloat(polygon[i + 1]));
    }
    // polygon is counterclockwise else convert
    if (sum < 0) {
        return polygon;
    } else {
        // flip array by pairs of points e.g. [x1, y1, x2, y2] -> [x2, y2, x1, y1]
        var result = [];
        for (var i = polygon.length - 2, j = 0; i >= 0; i -= 2, j += 2) {
            result[j] = polygon[i]; // x val
            result[j + 1] = polygon[i + 1]; // y val
        }
    }
    return result
}

function toClockwise(polygon) {
    var sum = 0;

    // loop through points and sum edges (x2-x1)(y2+y1)
    for (var i = 0; i + 3 < polygon.length; i += 2) {
        sum += (parseFloat(polygon[i + 2]) - parseFloat(polygon[i])) * (parseFloat(polygon[i + 3]) + parseFloat(polygon[i + 1]));
    }

    // polygon is counterclockwise else convert
    if (sum >= 0) {
        return polygon; s
    } else {
        // flip array by pairs of points e.g. [x1, y1, x2, y2] -> [x2, y2, x1, y1]
        var result = [];
        for (var i = polygon.length - 2, j = 0; i >= 0; i -= 2, j += 2) {
            result[j] = polygon[i]; // x val
            result[j + 1] = polygon[i + 1]; // y val
        }
    }
    return result
}

//Parses string from _background field, translates the vertex using the geometric mean, then
//stores vertex as a point computed from the lambertAzimuthal projection.
var polygonStrToHyperbolic = function(xStr,yStr){
  x = parseFloat(xStr);
  y = parseFloat(yStr);

  node = new Node(x-geoMean[0],y-geoMean[1]);

  return(lambertAzimuthal(node.r,node.theta));

}

var preserveOriginalMapNode = function(xStr,yStr){
  x = parseFloat(xStr);
  y = parseFloat(yStr);

  node = new Node(x-geoMean[0],y-geoMean[1]);

  return(node);

}

//Takes as input a polygon, returns the polygon transformed about the new origin.
var transformPolygon = function(P,newOrigin,zoom){

  vertices = P.getVertices();
  for (i in vertices){
    hR = vertices[i].getHyperbolicRadius()
    theta = vertices[i].getAngle();
    vertices[i] = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,theta);

    vertices[i] = mobius(vertices[i],newOrigin);
  }

  return(HyperbolicCanvas.Polygon.givenVertices(vertices))
}

var mobius = function(z,transform){
  z0 = math.Complex.fromPolar(z.getEuclideanRadius(),z.getAngle());

  numerator = math.multiply(transform.a,z0);
  numerator = math.add(numerator,transform.b);
  denominator = math.multiply(transform.c,z0);
  denominator = math.add(denominator,transform.d);

  newPoint = math.divide(numerator,denominator).toPolar();
  return(HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(
    newPoint.r,
    newPoint.phi)
  );
}

var parse_pos = function(strPos){
  Coords = strPos.split(',');
  return([parseFloat(Coords[0]),parseFloat(Coords[1])]);
}



class Node {
  constructor(x,y){
    this.x = x*SCALE_FACTOR;
    this.y = y*SCALE_FACTOR;
    this.r = this.calcR();
    this.theta = this.calcTheta();
  }

  calcR(){
    return (Math.sqrt(this.x*this.x + this.y*this.y))
  }

  calcTheta(){
    if (this.r === 0){
      console.log("undefined radius");
      return NaN
    }
    if(this.y >= 0){
      return Math.acos(this.x/this.r);
    }
    else {
      return -1*Math.acos(this.x/this.r);
    }
  }

  getR(){
    return (this.r)
  }

}

var lambertAzimuthal = function(r,theta){
  hR = math.acosh((.5*r*r)+1);

  return(HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,theta));
}

var makeMap = function(t){
  var regions;

  var color;
  var polygonsIdx = 0;
  var colorIdx = 0;
  var lineIdx = 0;
  let colors = [];
  let polygons = [[]];
  let lines = [[]];


  //Parsing code taken from http://gmap.cs.arizona.edu
  regions = t.graph._background.trim().split(/\s+/);
  // parse xdot for region info
  for (var i = 0; i < regions.length; i++) {
      if (regions[i] == "c") { // following specifies color
          i += 2;
          colors[colorIdx] = regions[i];

          if (colors[colorIdx].charAt(0) == '-') { // some color hex's have '-' before
              colors[colorIdx] = colors[colorIdx].substring(1);
          }
          colorIdx++;

      } else if (regions[i] == "P") { // following is a polygon
          i++;
          var size = parseInt(regions[i]); // number of points in polygon

          var polygon = regions.slice(i + 1, i + 1 + size * 2);

          polygon = toCounterClockwise(polygon); // this many dimensions for GeoJson polygon coordinates
          polygons[polygonsIdx++] = polygon;
      } else if (regions[i] == "L") { // following is a line border of the polygon
          i++;
          var size = parseInt(regions[i]);

          var line = regions.slice(i + 1, i + 1 + size * 2);
          lines[lineIdx++] = line;
      }
  }

  if (polygons.length > lines.length)
     l = polygons.length;
  else
     l = lines.length;

if (typeof polygons[0] != "undefined"){
myPolygons = [];
preservePolygons = [];
for(i=0; i<polygons.length; i++){
 myPolygons.push([]);
 preservePolygons.push([]);
 for (j=0; j < polygons[i].length; j+=2){
   myPolygons[i].push(polygonStrToHyperbolic(polygons[i][j],polygons[i][j+1]));
   preservePolygons[i].push(preserveOriginalMapNode(polygons[i][j],polygons[i][j+1]));
 }
}
}

if(typeof lines[0] != "undefined") {
myLines = [];
preserveLines = [];
for(i=0; i<lines.length; i++){
 myLines.push([]);
 preserveLines.push([]);
 for (j=0; j < lines[i].length; j+=2){
   myLines[i].push(polygonStrToHyperbolic(lines[i][j],lines[i][j+1]));
   preserveLines[i].push(preserveOriginalMapNode(lines[i][j],lines[i][j+1]));
 }
}
}

console.log(typeof polygons[0])
polygonList = [];
mapNodes = [];
if (typeof lines[0][0] != "undefined"){
  console.log('lines are used')
  console.log(lines)
  for(i = 0; i<myLines.length; i++){
   polygonList.push(HyperbolicCanvas.Polygon.givenVertices(myLines[i]));
   mapNodes.push(preserveLines[i]);
  }
}
else if (typeof polygons[0] != "undefined" && polygons[0].length > 0){
  console.log('polygons are used')
  for(i = 0; i<myPolygons.length; i++){
   polygonList.push(HyperbolicCanvas.Polygon.givenVertices(myPolygons[i]));
   mapNodes.push(preservePolygons[i]);
  }
}


console.log(polygonList)

return({
  polygonList: polygonList,
  colors: colors
});

}

var makeGraph = function(V,E){
  let i;
  let name;

  let nodeList = [];



  for (name in V){
    pos = parse_pos(V[name].pos);
    V[name].node = new Node(pos[0]-geoMean[0],pos[1]-geoMean[1]);
    V[name].hPos = lambertAzimuthal(V[name].node.r,V[name].node.theta);
    if (V[name].label && V[name].label != "\\N" ){
      V[name].labelPos = {
        name: V[name].label,
        nameLoc: new Node(pos[0],pos[1])
      }
    }
    else{
      V[name].labelPos = {
        name: name,
        nameLoc: new Node(pos[0],pos[1])
      }
    }
  nodeList.push(V[name])
  }


  let pathList = [];
  for (i in E){
    pathList.push([V[E[i].v],V[E[i].w]]);
  }

  return {
    nodeList,
    pathList
  }

}

var compute_geometric_mean = function(V) {
  let allX = 0;
  let allY = 0;
  let count = 0;

  for (name in V){
    pos = parse_pos(V[name].pos);
    allX += pos[0];
    allY += pos[1];
    count += 1;
  }
  return [allX/count, allY/count]
}

var newMobius = function(z,transform){
  numerator = math.multiply(z,transform.a)
  numerator = math.add(numerator,transform.b)

  denominator = math.multiply(z,transform.c)
  denominator = math.add(denominator,transform.d)

  fraction = math.divide(numerator,denominator)
  return(fraction)
}

//Runs on first call
HyperbolicCanvas.scripts['poincare_disk'] = function (canvas,graphStr) {
    let n = 0;
    let is_line_set = false;
    let currentlyRecentering = false;
    let previousCoverage, currentCoverage, previousZoom, currentZoom, destination;

    t = graphStr

    V = {}
    E = {}
    originalMapNodes = [];

    //Bit of a hack to convert the Gmap graph data to what the implementation in makeGraph expects.
    for (i in t.nodes){
      if(! t.nodes[i].id.includes('dummy')){
        v = t.nodes[i].id
        V[v] = {}
        V[v].pos = t.nodes[i].pos
        V[v].label = t.nodes[i].label
        if(t.nodes[i].color){
          V[v].color = t.nodes[i].color;
        }
      }
    }
    for(i in t.links){
      E[i] = {v: t.links[i].source,
              w: t.links[i].target}
    }

    //Computes the geometric mean of the nodes so we can make the mean the origin.
    geoMean = compute_geometric_mean(V);

    G = makeGraph(V,E);

    console.log("Nodelist is")
    console.log(G.nodeList);

    //If there is any background data, parse it.
    if(t.graph._background){

      Map = makeMap(t);

      //Check if this is a line set.
      if(t.graph._background.slice(0,100).includes("-setlinewidth(15)")){
        is_line_set = true;
      }
    }

    startLocations = [];
    for (i in G.nodeList){
      startLocations.push(G.nodeList[i].hPos);
    }
    startMapLocation = [];
    for (i in Map.polygonList){
      startMapLocation.push(Map.polygonList[i]);
    }

    console.log($("#hyperbolic-canvas")[0]);
    console.log(G.nodeList[3].node)

    //Set various default canvas properties
    var ctx = canvas.getContext();

    var defaultProperties = {
      lineJoin: 'round',
      lineWidth: 1,
      strokeStyle: "#D3D3D3",
      fillStyle: '#66B2FF',
      textAlign: 'center'
    }

    canvas.setContextProperties(defaultProperties);

    console.log(canvas._setupSize);

    var location = HyperbolicCanvas.Point.givenCoordinates(0,0);
    var initialLocation = HyperbolicCanvas.Point.givenCoordinates(0,0);
    let keepGoing = true;
    destination = HyperbolicCanvas.Point.givenCoordinates(0,0);

    //Called every animation frame.
    var render = function (event) {
      canvas.clear();
      keepGoing = true;

      if(currentlyRecentering){
        step = location.euclideanDistantPoint(.01,destination);
        if(step.euclideanDistanceTo(destination) < .01){
          currentlyRecentering = false;
          changeCenter(destination);
          location = HyperbolicCanvas.Point.givenCoordinates(0,0);
        }else{
          changeCenter(step);
          location = HyperbolicCanvas.Point.givenCoordinates(0,0);
        }
      }

      if($("#resetMap")[0].value === "NeedsReset"){
        changeCenter(initialLocation);
        location = HyperbolicCanvas.Point.givenCoordinates(0,0);
        $("#resetMap")[0].value = "Reset";

      }

      currentCoverage = $('#Coverage')[0].value;
      currentZoom = $('#Zoom')[0].value;

      if(currentZoom !== previousZoom){
        document.getElementById('hyperbolic-canvas').style.height = $('#Zoom')[0].value+'%';
        canvas.clear();
        canvas.setupSize2();
      }

      if(currentCoverage !== previousCoverage){
        scaleFactor = (parseFloat($('#Coverage')[0].value))*.01;
        keepGoing = false;

        initR = initialLocation.getEuclideanRadius();
        initTheta = initialLocation.getAngle();
        z0 = math.Complex.fromPolar(initR,initTheta);
        transform = {
          a: -1,
          b: math.Complex.fromPolar(initR,initTheta).neg(),
          c: math.Complex.fromPolar(initR,initTheta).conjugate().neg(),
          d: -1
        };


        for(i in G.nodeList){
          //r = startLocations[i].getHyperbolicRadius();
          //theta = startLocations[i].getAngle();
          newR = G.nodeList[i].node.r * scaleFactor;
          tempPos = lambertAzimuthal(newR,G.nodeList[i].node.theta);
          //tempPos = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(r+scaleFactor,theta);
          G.nodeList[i].hPos = mobius(tempPos,transform);
        }

        for (i in mapNodes){
          newVertices = []
          for (j in mapNodes[i]){
            newR = mapNodes[i][j].r * scaleFactor;
            tempPos = lambertAzimuthal(newR,mapNodes[i][j].theta);
            newVertices.push(mobius(tempPos,transform));
          }
          Map.polygonList[i] = HyperbolicCanvas.Polygon.givenVertices(newVertices);
        }


      }

      previousCoverage = currentCoverage;
      previousZoom = currentZoom;


      //Draw map background.
      ctx.strokeStyle = 'black';
      for(i in Map.polygonList){
        ctx.fillStyle = Map.colors[i];
        path = canvas.pathForHyperbolic(Map.polygonList[i]);
        if(is_line_set){
          ctx.strokeStyle = Map.colors[i];
          ctx.lineWidth = 15;
          canvas.stroke(path);
        }else{
          canvas.fillAndStroke(path);
        }
      }

      //Draw edges
      ctx.globalAlpha = parseFloat($("#transparency")[0].value)*.01; //Transparency value
      ctx.lineWidth = 1;
      for(i in G.pathList){
        ctx.strokeStyle = "grey";
        path = canvas.pathForHyperbolic(HyperbolicCanvas.Line.givenTwoPoints(
          G.pathList[i][0].hPos,
          G.pathList[i][1].hPos
        ));
        canvas.stroke(path);
      }
      ctx.globalAlpha = 1;

      //Draw nodes and place text
      nodesAllowed = parseFloat($("#nodesVisible")[0].value)*1;
      for(i in G.nodeList){
        ctx.fillStyle = "grey";
        if(G.nodeList[i].color){
          ctx.fillStyle = G.nodeList[i].color
        }
        path = canvas.pathForHyperbolic(
          HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(G.nodeList[i].hPos,.05)
        );
        canvas.fillAndStroke(path);

        ctx.fillStyle = "black";
        nodeDis = G.nodeList[i].hPos.getEuclideanRadius();

        //Scale text by distance from origin
        if( nodeDis < nodesAllowed){
          fontSize = Math.ceil(nodesAllowed *(1-nodeDis));
          ctx.font = fontSize.toString() + "px Arial";

          let pixelCoord = canvas.getCanvasPixelCoords(G.nodeList[i].hPos);
          ctx.fillText(G.nodeList[i].labelPos.name, pixelCoord[0],pixelCoord[1]);
        }
      }

      requestAnimationFrame(render);
    };

//Transforms the graph (and map) about a new origin, center.
var changeCenter = function(center,zoom=1){
  canvas.clear();
  r = center.getEuclideanRadius();
  theta = center.getAngle();

  transform = {
    a: 1,
    b: math.Complex.fromPolar(r,theta).neg(),
    c: math.Complex.fromPolar(r,theta).conjugate().neg(),
    d: 1
  };

  //VERY important to remember
  location = mobius(location,transform);
  initialLocation = mobius(initialLocation,transform);
  destination = mobius(destination,transform);

  for (i in Map.polygonList){
    Map.polygonList[i] = transformPolygon(Map.polygonList[i],transform,zoom);
  }

  for(i in G.nodeList){
    hR = G.nodeList[i].hPos.getHyperbolicRadius();
    theta = G.nodeList[i].hPos.getAngle();
    node = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,theta);

    G.nodeList[i].hPos = mobius(node,transform);
  }

}

//Finds how far the mouse has traveled and calls changeCenter on a new origin
//of that distance from the previous.
var resetLocation = function (event) {
  if (event) {
    x = event.clientX;
    y = event.clientY;
  }
  distance = [(translateX-x)*SCROLL_SPEED,(translateY-y)*SCROLL_SPEED];

  location = canvas.getCanvasPixelCoords(location);
  location = canvas.at([location[0] + distance[0], location[1] + distance[1]]);

  changeCenter(location);

};

var double_click = function(e){
  var boundBox = document.getElementById('hyperbolic-canvas').children[0]
  var rect = boundBox.getBoundingClientRect();
  //Convert to cartesian coordinates
  newX = ((e.clientX - rect.left)/(rect.right-rect.left)-0.5)*2;
  newY = ((e.clientY - rect.top)/(rect.bottom-rect.top)-0.5)*-2;

  //Ensure point is on the unit disk, then recenter on it.
  if(newX*newX + newY*newY <= 1){
    newPoint = HyperbolicCanvas.Point.givenCoordinates(newX,newY);
    //changeCenter(newPoint);
    //location = HyperbolicCanvas.Point.givenCoordinates(0,0)
    destination = newPoint;
    currentlyRecentering = true;
  }
}

var mouse_down = function(e){
  dragged = true;
  translateX = e.clientX;
  translateY = e.clientY;
  //console.log(canvas.at([translateX,translateY]))
  //changeCenter(canvas.at([.2,.2]))

}
var mouse_up = function(e){
  dragged = false;
  translateX = 0;
  translateY = 0;
}

var whileDragging = function(e){
  if(dragged){
    resetLocation(e);
  }
}

var changeZoom = function(zoom){
  console.log("i have made it here")
  for (i in G.nodeList){
    angle = G.nodeList[i].hPos.getAngle();
    hR = G.nodeList[i].hPos.getHyperbolicRadius()+zoom*0.1;
    G.nodeList[i].hPos = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,angle);
  }
}

var scroll = function(e){
  /*zoom = e.deltaY*.01;
  totalZoom = totalZoom + e.deltaY*.01;
  if(totalZoom < 300 && totalZoom >-300){
    console.log('here now')
    changeZoom(zoom);
  }
  else{
    totalZoom = totalZoom - e.deltaY*.01;
  }*/
}


    //canvas.getCanvasElement().addEventListener('click', incrementN);
    canvas.getCanvasElement().addEventListener('mousemove', whileDragging);
    canvas.getCanvasElement().addEventListener('wheel', scroll);
    canvas.getCanvasElement().addEventListener('mousedown', mouse_down);
    canvas.getCanvasElement().addEventListener('mouseup', mouse_up);
    canvas.getCanvasElement().addEventListener('dblclick',double_click);



    requestAnimationFrame(render);
  };
})();
