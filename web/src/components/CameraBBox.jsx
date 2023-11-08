import { Fragment, h } from 'preact';
import ActivityIndicator from './ActivityIndicator';
import { useApiHost } from '../api';
import useSWR from 'swr';
import { useCallback, useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { useResizeObserver } from '../hooks';

export default function CameraBBox({ camera, onload, searchParams = '', stretch = false }) {
  const { data: config } = useSWR('config');
  const apiHost = useApiHost();
  const [hasLoaded, setHasLoaded] = useState(false);
  const [bBoxList, setBBoxList] = useState([]);
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [{ width: containerWidth }] = useResizeObserver(containerRef);

  // Add scrollbar width (when visible) to the available observer width to eliminate screen juddering.
  // https://github.com/blakeblackshear/frigate/issues/1657
  let scrollBarWidth = 0;
  if (window.innerWidth && document.body.offsetWidth) {
    scrollBarWidth = window.innerWidth - document.body.offsetWidth;
  }
  const availableWidth = scrollBarWidth ? containerWidth + scrollBarWidth : containerWidth;

  const { name } = config ? config.cameras[camera] : '';
  const enabled = config ? config.cameras[camera].enabled : 'True';
  const { width, height } = config ? config.cameras[camera].detect : { width: 1, height: 1 };
  const aspectRatio = width / height;

  const scaledHeight = useMemo(() => {
    const scaledHeight = Math.floor(availableWidth / aspectRatio);
    const finalHeight = stretch ? scaledHeight : Math.min(scaledHeight, height);

    if (finalHeight > 0) {
      return finalHeight;
    }

    return 100;
  }, [availableWidth, aspectRatio, height, stretch]);
  const scaledWidth = useMemo(
    () => Math.ceil(scaledHeight * aspectRatio - scrollBarWidth),
    [scaledHeight, aspectRatio, scrollBarWidth]
  );
  //-----
  const fetchBBox = async () => {
    try {
      const response = await fetch(`${apiHost}api/${name}/latest_results?${searchParams ? `&${searchParams}` : ''}`);
      const bboxListResponse = await response.json();
      setBBoxList(bboxListResponse['objects'])
    } catch (error) {
      console.error('Error:', error);
    }
  }

  useEffect(() => {
    if (!config || scaledHeight === 0 || !canvasRef.current) {
      return;
    }
    fetchBBox();
  }, [apiHost, canvasRef, name, searchParams, scaledHeight, config]);

  useEffect(() => {
    setHasLoaded(true);
    if (canvasRef.current) {
      // console.log(bBoxList)
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      bBoxList.forEach(item => {
        const scaledX = Math.round(item['box'][0] / width * scaledWidth);
        const scaledY = Math.round(item['box'][1] / height * scaledHeight);
        const scaledBoxW = Math.round((item['box'][2] - item['box'][0]) / width * scaledWidth);
        const scaledBoxH = Math.round((item['box'][3] - item['box'][1]) / height * scaledHeight);
        ctx.strokeRect(scaledX, scaledY, scaledBoxW, scaledBoxH);
      });
      // ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);
      onload && onload();
    }
  }, [bBoxList, setHasLoaded, canvasRef])

  return (
    <Fragment>
      <div className="absolute z-3 w-full h-full left-0 top-0 flex justify-center" ref={containerRef}>
        {enabled ? (
          <canvas data-testid="cameraimage-canvas" height={scaledHeight} ref={canvasRef} width={scaledWidth} />
          // <div className="absolute z-3 w-full h-full left-0 top-0" ref={svgRef} >
          //   <svg width="100%" height="100%" viewBox={'0 0 ' + width + ' ' + height}>
          //     {
          //       bboxList.map((item) => (
          //         <rect x={item['box'][0]} y={item['box'][1]} width={item['box'][2] - item['box'][0]} height={item['box'][3] - item['box'][1]}
          //           fill-opacity="0" stroke={`rgb(${item['color'][2]},${item['color'][1]},${item['color'][0]})`} stroke-width={item['thickness'] + 3} />
          //       ))
          //     }
          //   </svg>
          // </div>
        ) : (
          <div class="text-center pt-6">Camera is disabled in config, no stream or snapshot available!</div>
        )}
        {!hasLoaded && enabled ? (
          <div className="absolute z-3 w-full h-full left-0 top-0 flex justify-center" ref={containerRef}>
            <ActivityIndicator />
          </div>
        ) : null}
      </div>
    </Fragment>
  );
}
